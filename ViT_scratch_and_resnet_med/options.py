# put argument related functions here
def build_parser(parser):
    parser.add_argument("--model-type", type=str, default="Base", choices=["Base", "Dino"], help="Preferred  Model Type")
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--savedir', default='./runs/16_32_100_1e-3_1e-8_Adam_0_25_3class/', help='path to save models')
    parser.add_argument('--max-epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-8, help='Min LR for ReduceLROnPlateau')
    parser.add_argument('--gpu', action='store_true', help='whether gpu is used')
    parser.add_argument('--cuda', type=str, default="0", help='cuda num')
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGDm"], help="optimizer name")
    parser.add_argument("--patience", type=int, default=25, help="Patience number for early stopping condition")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay ") # default=1e-6
    parser.add_argument("--dataset-type", type=str, default="3class", choices=["3class", "5class"], help="which dataset is going to be used (3 or 5 classes)")
    parser.add_argument("--percentage", type=float, default=0.95, help="Used percentage of train data")
    return parser

