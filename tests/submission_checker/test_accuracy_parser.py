import pytest
from submission_checker.parsers.accuracy_parser import parse_line


# ---------------------------------------------------------------------------
# Regex-backed metrics
# ---------------------------------------------------------------------------

class TestAccMetric:
    def test_plain_accuracy_line(self):
        assert parse_line("accuracy = 76.50", "acc") == pytest.approx(76.50)

    def test_json_style_accuracy_line(self):
        assert parse_line('{"accuracy": 76.50}', "acc") == pytest.approx(76.50)

    def test_no_match_returns_none(self):
        assert parse_line("something else entirely", "acc") is None


class TestAUCMetric:
    def test_auc_line(self):
        assert parse_line("AUC=80.31", "AUC") == pytest.approx(80.31)

    def test_auc_with_trailing_text(self):
        assert parse_line("AUC=80.31 (threshold=0.5)", "AUC") == pytest.approx(80.31)

    def test_no_match_returns_none(self):
        assert parse_line("accuracy = 80.31", "AUC") is None


class TestMAPMetric:
    def test_map_equals_format(self):
        assert parse_line("mAP=37.55", "mAP") == pytest.approx(37.55)

    def test_map_total_dict_format(self):
        assert parse_line("'Total': 37.55", "mAP") == pytest.approx(37.55)

    def test_no_match_returns_none(self):
        assert parse_line("Average Precision = 37.55", "mAP") is None


class TestACCURACYMetric:
    def test_wer_accuracy_line(self):
        val = parse_line("Word Error Rate: 4.5%, accuracy=95.5%", "ACCURACY")
        assert val == pytest.approx(95.5)

    def test_no_match_returns_none(self):
        assert parse_line("accuracy=95.5%", "ACCURACY") is None


class TestDICEMetric:
    def test_dice_line(self):
        assert parse_line("Accuracy: mean = 0.86170", "DICE") == pytest.approx(0.86170)

    def test_no_match_returns_none(self):
        assert parse_line("mean accuracy 0.86", "DICE") is None


class TestDLRMMetrics:
    def test_dlrm_ne(self):
        val = parse_line("metric/lifetime_ne/rating: 0.8500", "DLRM_NE")
        assert val == pytest.approx(0.85)

    def test_dlrm_acc(self):
        val = parse_line("metric/lifetime_accuracy/rating: 0.9200", "DLRM_ACC")
        assert val == pytest.approx(0.92)

    def test_dlrm_auc(self):
        val = parse_line("metric/lifetime_gauc/rating: 0.8100", "DLRM_AUC")
        assert val == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# Dict-backed metrics (ast.literal_eval)
# ---------------------------------------------------------------------------

class TestROUGEMetrics:
    ROUGE_LINE = "{'rouge1': 44.43, 'rouge2': 22.04, 'rougeL': 28.62, 'rougeLsum': 35.0, 'gen_len': 8167644}"

    def test_rouge1(self):
        assert parse_line(self.ROUGE_LINE, "ROUGE1") == pytest.approx(44.43)

    def test_rouge2(self):
        assert parse_line(self.ROUGE_LINE, "ROUGE2") == pytest.approx(22.04)

    def test_rougel(self):
        assert parse_line(self.ROUGE_LINE, "ROUGEL") == pytest.approx(28.62)

    def test_rougelsum(self):
        assert parse_line(self.ROUGE_LINE, "ROUGELSUM") == pytest.approx(35.0)

    def test_gen_len(self):
        assert parse_line(self.ROUGE_LINE, "GEN_LEN") == pytest.approx(8167644)

    def test_no_dict_returns_none(self):
        assert parse_line("rouge1 = 44.43", "ROUGE1") is None


class TestTokensPerSample:
    def test_tokens_per_sample(self):
        line = "{'tokens_per_sample': 294.45}"
        assert parse_line(line, "TOKENS_PER_SAMPLE") == pytest.approx(294.45)


class TestCLIPAndFIDMetrics:
    CLIP_LINE = "Accuracy Results: {'CLIP_SCORE': 31.69, 'FID_SCORE': 23.01}"

    def test_clip_score(self):
        assert parse_line(self.CLIP_LINE, "CLIP_SCORE") == pytest.approx(31.69)

    def test_fid_score(self):
        assert parse_line(self.CLIP_LINE, "FID_SCORE") == pytest.approx(23.01)

    def test_clip_score_missing_prefix_returns_none(self):
        assert parse_line("{'CLIP_SCORE': 31.69}", "CLIP_SCORE") is None


# ---------------------------------------------------------------------------
# JSON-backed metrics
# ---------------------------------------------------------------------------

class TestF1Metric:
    def test_f1_line(self):
        assert parse_line('{"f1": 90.874}', "F1") == pytest.approx(90.874)

    def test_f1_with_prefix(self):
        assert parse_line('prefix text {"f1": 90.874}', "F1") == pytest.approx(90.874)

    def test_f1_hierarchical(self):
        assert parse_line('{"f1": 85.0}', "F1_HIERARCHICAL") == pytest.approx(85.0)

    def test_no_json_returns_none(self):
        assert parse_line("f1 = 90.874", "F1") is None

    def test_missing_key_returns_none(self):
        assert parse_line('{"score": 90.874}', "F1") is None


# ---------------------------------------------------------------------------
# Unknown metric
# ---------------------------------------------------------------------------

def test_unknown_metric_returns_none():
    assert parse_line("accuracy = 75.0", "UNKNOWN_METRIC") is None
