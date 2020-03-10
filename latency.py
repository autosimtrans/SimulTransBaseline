import al
from IPython import embed
import sys

pred_file_name, src_file_name = sys.argv[1], sys.argv[2]

# with open('decode/ft.zh-en.dec.114k.w1.b1.en.stream.detok.detc.unbpe', 'r') as f:
# with open('decode/wmt18.zh-en.dec.100k.w-1.b1.en.stream.unbpe.detok.detc', 'r') as f:
with open(pred_file_name, 'r', encoding='UTF-8') as f:
    tgt_lines = []
    for line in f.readlines():
        # exclude '\n'
        tgt_lines.append(line[:-1])


# with open('/mnt/scratch/zrenj/Project/challenge/transformer/data/zh-en.dev.zh.cut.json', 'r') as f:
# with open('/mnt/scratch/zrenj/Project/challenge/transformer/data/zh-en.dev.zh.json', 'r') as f:
with open(src_file_name, 'r', encoding='UTF-8') as f:
    import json
    src_lines = json.load(f)


rws = []
als = []
idx = 0
with open(pred_file_name+'.merge', 'w') as f:
    for talk in src_lines:
        sent = ''
        for sentence in talk:
            _rw = []
            words = []
            for part_sent in sentence:
                crt_sent = tgt_lines[idx]
                # apply a read
                if (len(crt_sent) > 0 and crt_sent[0] != ' ') and (len(sent) > 0 and sent[-1] != ' '):
                    # it's a bpe or detokenized word
                    # print('%s|%s'%(sent, crt_sent))
                    for i in range(len(_rw))[::-1]:
                        if _rw[i] == 1:
                            _rw[i] = 0
                            break
                else:
                    _rw += [0]
                sent += crt_sent
                for w in crt_sent.split():
                    # apply a write
                    _rw += [1]

                idx += 1
            rws.append(_rw)

            # print(' '.join(words), end='')
            # f.write(sent+' ')
            # f.write(sent)
            _ap, _cw, _al = al.delay(rws[-1])
            als.append(_al)
            # embed()
        f.write(sent+'\n', encoding='UTF-8')

assert(idx == len(tgt_lines))
print('AL:', sum(als) / len(als))
# embed()

