import hydra
import wandb
import time
from hydra.utils import instantiate
from math import ceil
from omegaconf import OmegaConf
from prettytable import PrettyTable

from datasets.cell.tabula_muris import *
from utils.io_utils import (
    get_resume_file,
    hydra_setup,
    fix_seed,
    model_to_dict,
    opt_to_dict,
    get_model_file,
)


def initialize_dataset_model(cfg):
    # Instantiate train dataset as specified in dataset config under simple_cls or set_cls
    if cfg.method.type == "baseline":
        train_dataset = instantiate(
            cfg.dataset.simple_cls, batch_size=cfg.method.train_batch, mode="train"
        )
    elif cfg.method.type == "meta":
        train_dataset = instantiate(cfg.dataset.set_cls, mode="train")
    else:
        raise ValueError(f"Unknown method type: {cfg.method.type}")
    train_loader = train_dataset.get_data_loader()

    # Instantiate val dataset as specified in dataset config under simple_cls or set_cls
    # Eval type (simple or set) is specified in method config, rather than dataset config
    if cfg.method.eval_type == "simple":
        val_dataset = instantiate(
            cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode="val"
        )
    else:
        val_dataset = instantiate(cfg.dataset.set_cls, mode="val")
    val_loader = val_dataset.get_data_loader()

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.method.name == "maml":
        cfg.method.stop_epoch *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    train_loader, val_loader, model = initialize_dataset_model(cfg)

    if cfg.method.name.startswith("bioMON"):
        run_bioMON(cfg, train_loader, val_loader, model)
    else:
        run_standard(cfg, train_loader, val_loader, model)

    print("Run complete! Updating wandb status...")


def run_standard(cfg, train_loader, val_loader, model):
    if cfg.mode == "train":
        model, training_time = train_standard(train_loader, val_loader, model, cfg)

    results = []
    print("Checkpoint directory:", cfg.checkpoint.dir)
    for split in cfg.eval_split:
        start_time = time.time()
        acc_mean, acc_std = test_standard(cfg, model, split)
        testing_time = time.time() - start_time
        results.append([split, acc_mean, acc_std, f"{testing_time:.2f}"])

    if not os.path.isdir(cfg.results.dir):
        os.makedirs(cfg.results.dir)
        
    test_summary_file = open(f"{cfg.results.dir}/test_summary.csv", "w")
    test_summary_file.write("split,acc_mean,acc_std,testing_time\n")
    for row in results:
        test_summary_file.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
    test_summary_file.close()

    print(f"Results logged to ./checkpoints/{cfg.exp.name}/results.txt")

    if cfg.mode == "train":
        table = wandb.Table(
            data=results, columns=["split", "acc_mean", "acc_std", "time(s)"]
        )
        wandb.log({"eval_results": table})

    display_table = PrettyTable(["split", "acc_mean", "acc_std", "time(s)"])
    for row in results:
        display_table.add_row(row)

    if cfg.mode == "train":
        print(f"Total training time: {training_time:.2f}s")

    print(display_table)


def run_bioMON(cfg, train_loader, val_loader, model):
    if cfg.mode == "train":
        (
            model,
            training_time,
            head_fit_time,
            head_support_acc,
            head_query_acc,
            avg_head_fit_time,
        ) = train_bioMON(train_loader, val_loader, model, cfg)

        print("Training complete! Logging the results...")
        display_table = PrettyTable(
            [
                "head_acc_support",
                "head_acc_query",
                "avg_head_fit_time (s)",
                "total_head_fit_time (s)",
                "total_training_time (s)",
            ]
        )
        display_table.add_row(
            [
                f"{head_support_acc:.2f}",
                f"{head_query_acc:.2f}",
                f"{avg_head_fit_time:.2f}",
                f"{head_fit_time:.2f}",
                f"{training_time:.2f}",
            ]
        )
        print(display_table)

    results = []
    detailed_results = []
    print("Checkpoint directory:", cfg.checkpoint.dir)
    for split in cfg.eval_split:
        start_time = time.time()
        (
            acc_mean,
            acc_std,
            head_support_acc_mean,
            head_support_acc_std,
            head_query_acc_mean,
            head_query_acc_std,
            head_fit_time_mean,
            head_fit_time_std,
            total_head_fit_time,
        ) = test_bioMON(cfg, model, split)
        testing_time = time.time() - start_time

        results.append(
            [
                split,
                acc_mean,
                acc_std,
                head_query_acc_mean,
                head_query_acc_std,
                total_head_fit_time,
                testing_time,
            ]
        )

        detailed_results.append(
            [
                split,
                acc_mean,
                acc_std,
                head_support_acc_mean,
                head_support_acc_std,
                head_query_acc_mean,
                head_query_acc_std,
                head_fit_time_mean,
                head_fit_time_std,
                total_head_fit_time,
                testing_time,
            ]
        )

    test_summary_file = open(f"{cfg.results.dir}/test_summary.csv", "w")
    test_summary_file.write(
        "split,acc_mean,acc_std,head_support_acc_mean,head_support_acc_std,head_query_acc_mean,head_query_acc_std,head_fit_time_mean,head_fit_time_std,total_head_fit_time,testing_time\n"
    )
    for row in detailed_results:
        test_summary_file.write(
            f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]},{row[10]}\n"
        )
    test_summary_file.close()

    print(f"Detailed results logged to {cfg.results.dir}/test_summary.csv")
    print(f"Single line results logged to ./checkpoints/{cfg.exp.name}/results.txt")

    if cfg.mode == "train":
        table = wandb.Table(
            data=results,
            columns=[
                "split",
                "acc_mean",
                "acc_std",
                "acc_head_query_mean",
                "acc_head_query_std",
                "head fit time (s)",
                "testing time (s)",
            ],
        )  # "test time(s)", "head fit time(s)", "head_test_acc_mean", "head_test_acc_std"])
        wandb.log({"eval_results": table})

    display_table = PrettyTable(
        [
            "split",
            "acc_mean",
            "acc_std",
            "acc_head_query_mean",
            "acc_head_query_std",
            "head fit time (s)",
            "testing time (s)",
        ]
    )
    for row in results:
        display_table.add_row(row)
    print(display_table)


def train_standard(train_loader, val_loader, model, cfg):
    cfg.checkpoint.time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # add short date and time to checkpoint dir
    # cfg.checkpoint.dir += f"/{cfg.checkpoint.time}"

    cp_dir = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.time)

    if not os.path.isdir(cp_dir):
        os.makedirs(cp_dir)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.exp.name,
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode,
    )
    wandb.define_metric("*", step_metric="epoch")

    if cfg.exp.resume:
        resume_file = get_resume_file(cp_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            cfg.method.start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])

    optimizer = instantiate(cfg.optimizer_cls, params=model.parameters())

    print("Model Architecture:")
    print(model)
    wandb.config.update({"model_details": model_to_dict(model)})

    print("Optimizer:")
    print(optimizer)
    wandb.config.update({"optimizer_details": opt_to_dict(optimizer)})

    max_acc = -1

    total_training_time = 0

    for epoch in range(cfg.method.start_epoch, cfg.method.stop_epoch):
        wandb.log({"epoch": epoch})
        model.train()

        training_start_time = time.time()
        model.train_loop(epoch, train_loader, optimizer)
        total_training_time += time.time() - training_start_time

        if epoch % cfg.exp.val_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            model.eval()
            acc = model.test_loop(val_loader)
            print(f"Epoch {epoch}: {acc:.2f}")
            wandb.log({"acc/val": acc})

            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(cp_dir, "best_model.tar")
                torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

        if epoch % cfg.exp.save_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            outfile = os.path.join(cp_dir, "{:d}.tar".format(epoch))
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    if not os.path.isdir(cfg.results.dir):
        os.makedirs(cfg.results.dir)
    
    train_summary_file = open(f"{cfg.results.dir}/train_summary.csv", "w")
    train_summary_file.write(
        "total_training_time,\n"
    )
    train_summary_file.write(
        f"{total_training_time},\n"
    )
    
    return model, total_training_time


def test_standard(cfg, model, split):
    if cfg.method.eval_type == "simple":
        test_dataset = instantiate(
            cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode=split
        )
    else:
        test_dataset = instantiate(
            cfg.dataset.set_cls, n_episode=cfg.iter_num, mode=split
        )

    test_loader = test_dataset.get_data_loader()

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)["state"])
    model.eval()

    if cfg.method.eval_type == "simple":
        acc_all = []

        num_iters = ceil(cfg.iter_num / len(test_dataset.get_data_loader()))
        cfg.iter_num = num_iters * len(test_dataset.get_data_loader())
        print("num_iters", num_iters)
        for i in range(num_iters):
            acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
            acc_all.append(acc_mean)

        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        # assert False, "Not implemented"
    else:
        # Don't need to iterate, as this is accounted for in num_episodes of set data-loader
        acc_mean, acc_std = model.test_loop(test_loader, return_std=True)

    with open(f"./checkpoints/{cfg.exp.name}/results.txt", "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = "%s-%s-%s-%s %sshot %sway" % (
            cfg.dataset.name,
            split,
            cfg.model,
            cfg.method.name,
            cfg.n_shot,
            cfg.n_way,
        )

        acc_str = "%4.2f%% +- %4.2f%%" % (
            acc_mean,
            1.96 * acc_std / np.sqrt(cfg.iter_num),
        )
        f.write(
            "Time: %s, Setting: %s, Acc: %s, Model: %s \n"
            % (timestamp, exp_setting, acc_str, model_file)
        )

    return acc_mean, acc_std


def train_bioMON(train_loader, val_loader, model, cfg):
    cfg.checkpoint.time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # add short date and time to checkpoint dir
    # cfg.checkpoint.dir += f"/{cfg.checkpoint.time}"

    cp_dir = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.time)

    if not os.path.isdir(cp_dir):
        os.makedirs(cp_dir)

    if not os.path.isdir(cfg.results.dir):
        os.makedirs(cfg.results.dir)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.exp.name,
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode,
    )
    wandb.define_metric("*", step_metric="epoch")

    if cfg.exp.resume:
        resume_file = get_resume_file(cp_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            cfg.method.start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])

    optimizer = instantiate(cfg.optimizer_cls, params=model.parameters())

    print("Model Architecture:")
    print(model)
    wandb.config.update({"model_details": model_to_dict(model)})

    print("Optimizer:")
    print(optimizer)
    wandb.config.update({"optimizer_details": opt_to_dict(optimizer)})

    max_acc = -1

    total_head_fit_time = 0
    total_training_time = 0

    all_head_train_acc = 0
    all_head_test_acc = 0

    train_results_file = open(f"{cfg.results.dir}/epoch_train_results.csv", "w")
    train_results_file.write(
        "epoch,train_loss,val_loss,val_acc,val_head_support_acc,val_head_query_acc\n"
    )

    for epoch in range(cfg.method.start_epoch, cfg.method.stop_epoch):
        wandb.log({"epoch": epoch})
        model.train()

        training_start_time = time.time()
        (
            train_loss,
            train_head_support_acc,
            train_head_query_acc,
            _,
            train_head_fit_time,
        ) = model.train_loop(epoch, train_loader, optimizer)
        total_training_time += time.time() - training_start_time

        total_head_fit_time += train_head_fit_time
        all_head_train_acc += train_head_support_acc
        all_head_test_acc += train_head_query_acc

        if epoch % cfg.exp.val_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            model.eval()
            (
                acc,
                head_support_acc,
                head_query_acc,
                head_fit_time,
                val_loss,
                _,
            ) = model.test_loop(val_loader)

            print(f"Epoch {epoch}: {acc:.2f}")
            wandb.log({"validation acc": acc})
            wandb.log({"validation support acc": head_support_acc})
            wandb.log({"validation query acc": head_query_acc})
            wandb.log({"validation fit time": head_fit_time})

            # Log training metrics
            # (epoch,train_loss,val_loss,val_acc,val_head_support_acc,val_head_query_acc)
            epoch_data = f"{epoch},{train_loss},{val_loss},{acc},{head_support_acc},{head_query_acc}\n"
            train_results_file.write(epoch_data)

            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(cp_dir, "best_model.tar")
                torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

        if epoch % cfg.exp.save_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            outfile = os.path.join(cp_dir, "{:d}.tar".format(epoch))
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    train_results_file.close()

    all_head_train_acc = all_head_train_acc / (
        cfg.method.stop_epoch - cfg.method.start_epoch
    )
    all_head_test_acc = all_head_test_acc / (
        cfg.method.stop_epoch - cfg.method.start_epoch
    )
    all_head_fit_time = total_head_fit_time / (
        cfg.method.stop_epoch - cfg.method.start_epoch
    )

    train_summary_file = open(f"{cfg.results.dir}/train_summary.csv", "w")
    train_summary_file.write(
        "total_training_time,total_head_fit_time,train_head_support_acc,train_head_query_acc,avg_head_fit_time\n"
    )
    train_summary_file.write(
        f"{total_training_time},{total_head_fit_time},{all_head_train_acc},{all_head_test_acc},{all_head_fit_time}\n"
    )

    train_summary_file.close()

    return (
        model,
        total_training_time,
        total_head_fit_time,
        all_head_train_acc,
        all_head_test_acc,
        all_head_fit_time,
    )


def test_bioMON(cfg, model, split):
    if cfg.method.eval_type == "simple":
        test_dataset = instantiate(
            cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode=split
        )
    else:
        test_dataset = instantiate(
            cfg.dataset.set_cls, n_episode=cfg.iter_num, mode=split
        )

    test_loader = test_dataset.get_data_loader()

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)["state"])
    model.eval()

    if cfg.method.eval_type == "simple":
        assert False, "BioMON is supposed to be evaluated with episodic data loader"
    else:
        # Don't need to iterate, as this is accounted for in num_episodes of set data-loader
        (
            acc_mean,
            acc_std,
            head_support_acc_mean,
            head_support_acc_std,
            head_query_acc_mean,
            head_query_acc_std,
            head_fit_time_mean,
            head_fit_time_std,
            loss_mean,
            loss_std,
            total_head_fit_time,
        ) = model.test_loop(test_loader, return_std=True)

    with open(f"./checkpoints/{cfg.exp.name}/results.txt", "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = "%s-%s-%s-%s %sshot %sway" % (
            cfg.dataset.name,
            split,
            cfg.model,
            cfg.method.name,
            cfg.n_shot,
            cfg.n_way,
        )

        acc_str = "%4.2f%% +- %4.2f%%" % (
            acc_mean,
            1.96 * acc_std / np.sqrt(cfg.iter_num),
        )
        f.write(
            "Time: %s, Setting: %s, Acc: %s, Model: %s \n"
            % (timestamp, exp_setting, acc_str, model_file)
        )

    return (
        acc_mean,
        acc_std,
        head_support_acc_mean,
        head_support_acc_std,
        head_query_acc_mean,
        head_query_acc_std,
        head_fit_time_mean,
        head_fit_time_std,
        total_head_fit_time,
    )


if __name__ == "__main__":
    hydra_setup()
    run()
    wandb.finish()
