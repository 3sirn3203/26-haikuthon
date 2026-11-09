import pytest

from haiku_bl import submission as S


def test_roundtrip_and_scoring(tmp_path):
    p = S.write_submission(tmp_path / 's.csv', ['0000', '0001'], [['a', 'b', 'c'], ['c', 'a']], top_k=2)
    assert S.read_submission(p) == {'0000': ['a', 'b'], '0001': ['c', 'a']}
    res = S.score_submission(p, {'0000': 'a', '0001': 'a', '0002': 'b'}, masks={'pub': ['0000']})
    assert res['pub']['score'] == 1.0
    assert res['all']['n'] == 3                        # missing row counts as 0
    assert res['all']['recall_at_1'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('ranked', [[], ['a', 'a'], ['a b']])
def test_invalid_rows(tmp_path, ranked):
    with pytest.raises(ValueError):
        S.write_submission(tmp_path / 's.csv', ['0'], [ranked])


def test_unknown_ids_and_duplicates(tmp_path):
    with pytest.raises(ValueError):
        S.write_submission(tmp_path / 's.csv', ['0'], [['zz']], image_ids={'a'})
    with pytest.raises(ValueError):
        S.write_submission(tmp_path / 's.csv', ['0', '0'], [['a'], ['a']])
