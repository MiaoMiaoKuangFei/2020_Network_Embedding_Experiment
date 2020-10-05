import optparse
import util

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-t', dest='t', help='Train-set', type='str', default='train.txt')
    parser.add_option('-s', dest='s', help='Test-set', type='str', default='test.txt')
    parser.add_option('-o', dest='o', help='Output', type='str', default='/result')
    parser.add_option('-m', dest='m', help='model', type='str', default='/emb/model')
    options, args = parser.parse_args()

    id2vec = util.import_model(options.m)

    # fixme：首先获取正例和负例，然后用cos距离填出一个list来，最后用label和cos距离来进行AUC的分数判定
