import itertools
import json
from pathlib import Path
import tempfile
import unittest

from paired_batches import make_paired_batches


class PairedBatchTests(unittest.TestCase):
    def test_order_does_not_depend_on_format_length_and_keeps_masks(self):
        with tempfile.TemporaryDirectory() as d:
            orders=[]
            for label,lengths in [('history',[33,65,97]),('compact',[111,45,9])]:
                path=Path(d)/label;rows=[(list(range(n)),2) for n in lengths]
                batches=list(itertools.islice(make_paired_batches(83,path)(rows,1,160,loop=True),3))
                visits=[json.loads(s) for s in path.read_text().splitlines()];orders.append([x['index'] for x in visits])
                for (tokens,spans),v in zip(batches,visits):
                    row=rows[v['index']];self.assertEqual(tokens[0,:len(row[0])].tolist(),row[0]);self.assertEqual(spans.tolist(),[[2,len(row[0])]])
            self.assertEqual(orders[0],orders[1]);self.assertEqual(sorted(orders[0]),[0,1,2])

    def test_matches_native_batch_padding_and_refuses_truncation(self):
        from mlx_lm.tuner.trainer import iterate_batches
        with tempfile.TemporaryDirectory() as d:
            rows=[(list(range(67)),8)]
            ours=next(make_paired_batches(83,Path(d)/'order')(rows,1,128))
            native=next(iterate_batches(rows,1,128))
            self.assertEqual(ours[0].tolist(),native[0].tolist());self.assertEqual(ours[1].tolist(),native[1].tolist())
            with self.assertRaises(ValueError):next(make_paired_batches(83,Path(d)/'order')(rows,1,64))


if __name__=='__main__':unittest.main()
