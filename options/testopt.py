import argparse

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Evaluate performance of SARPN on NYU-D v2 test set')
    parser.add_argument('--testlist_path', required=True, help='the path of testlist')
    parser.add_argument('--root_path', required=True, help="the root path of dataset")
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--loadckpt', required=True, help="the path of the loaded model")
    # parse arguments
    return parser.parse_args()
