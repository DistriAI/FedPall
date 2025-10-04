import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--exp', type = int, default=6, help ='exp number(default:1)')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication(default:300)')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication(default:1")')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size(default:16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--number_workers', type = int, default=4, help = 'dataloader number workers(default:0)')
    parser.add_argument('--size', type = int, default= 64, help ='images input size(default:256)')
    parser.add_argument('--mode', type = str, default='ours', help='fedavg | fedprox | fedBN | SingleSet | fedproto')
    parser.add_argument('--dataset', type = str, default='digit', help='digit | domainnet | office | PACS')
    parser.add_argument('--optimizer', type = str, default='sgd', help='sgd | adam')
    parser.add_argument('--num_users', type=int, default=5, help ='the number of clients(default:5)')
    parser.add_argument('--features', type=int, default=1, help ='0 denotes non-iid features, 1 denotes iid features')
    parser.add_argument('--labels', type=int, default=0, help ='0 denotes non-iid labels, 1 denotes iid labels')
    parser.add_argument('--num_classes', type=int, default=10, help ='class numbers')
    parser.add_argument('--save_path', type = str, default=None, help='path to save the outcome')
    parser.add_argument('--device', default="cuda:0", type=str, help="cpu, cuda, or others")
    # ablation
    parser.add_argument('--momentum', type = float, default=1, help ='upper momentum of our algorithm')
    parser.add_argument('--loss_component', type=int, default=1, help ='loss combination')
    parser.add_argument('--cls_component', type=int, default=1, help ='classification change strategy')
    parser.add_argument('--mix_mode', type=int, default=1, help ='noise type')
    # dataset
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train for digits')
    # algorithm
    # ours
    parser.add_argument('--adcol_epoch', type = int, default=3, help = 'optimization iters in discriminator(default:3")')
    parser.add_argument('--uniform_left', type=float, default=0.8)
    parser.add_argument('--uniform_right', type=float, default=0.2)
    parser.add_argument('--adcol_mu', type=float, default=0.1, help='The hyper parameter for fedprox')
    parser.add_argument('--adcol_beta', type=float, default=0.1, help='The hyper parameter for fedprox')
    # others
    parser.add_argument('--feat_loss_arg', type=float, default=0.1)
    parser.add_argument('--crt_feat_num', type=int, default=50)
    parser.add_argument('--ld', type = float, default= 1, help ='weight for fedproto mse loss')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--alpha', default=1, type=float, help='the hypeparameter for BKD2 loss')
    parser.add_argument('--T', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--beta', type=float, default=1e-2, help ='beta for perfedavg')
    parser.add_argument('--tau', type=float, default=1, help ='temperature coefficient')
    parser.add_argument('--times', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=100, help="CE loss weight parameter")
    # fedUV
    parser.add_argument('--std_coeff', type=float, default=2.5, help='the lambda parameter for FedUV')
    parser.add_argument('--unif_coeff', type=float, default=0.5, help='the mu parameter for FedUV')

    args = parser.parse_args()
    return args

