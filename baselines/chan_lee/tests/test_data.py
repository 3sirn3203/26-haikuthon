"""Split logic on the real prompt/component files (no images needed)."""
import pytest

from haiku_bl import data as D


@pytest.fixture(scope='module')
def cfg():
    return D.load_paths()


def test_bundles_match_stage1_design(cfg):
    for split, easy, n in (('train', 400, 450), ('test', 800, 900)):
        b = D.bundles(split, cfg)
        assert sum(v is None for v in b.values()) == easy
        sizes = {}
        for v in b.values():
            if v is not None:
                sizes[v] = sizes.get(v, 0) + 1
        assert len(sizes) == n and set(sizes.values()) == {8}


def test_val_split_is_bundle_disjoint(cfg):
    fit, val = D.val_split(cfg)
    assert len(val) == cfg['val_size'] and not fit & val and len(fit | val) == 4000
    b = D.bundles('train', cfg)
    val_bundles = {b[p] for p in val if b[p]}
    assert not any(b[p] in val_bundles for p in fit)


def test_eval_sets(cfg):
    test = D.build_eval_set('test', cfg)
    assert len(test.queries) == 8000
    pub = [q for q in test.queries if 'public' in q.tags]
    assert len(pub) == cfg['public_size']
    hn = [im for im in test.pool if im.kind == 'hn']
    assert len(hn) == 8000 - len(pub)                     # public queries' HN excluded
    assert not {im.source_pid for im in hn} & {q.pid for q in pub}
    assert (test.gt_index < 8000).all()
    val = D.build_eval_set('val', cfg)
    assert len(val.queries) == cfg['val_size']
    assert {im.source_pid for im in val.pool if im.kind == 'hn'} <= {q.pid for q in val.queries}
