# end-to-end evaluation
.PHONY: all
all: resnet retinanet 3d-unet bert rnnt gpt-j

.PHONY: resnet
resnet:
	-bash vision/classification_and_detection/run_eval_resnet.sh

.PHONY: retinanet
retinanet:
	-bash vision/classification_and_detection/run_eval_retinanet.sh

.PHONY: 3d-unet
3d-unet:
	-bash vision/medical_imaging/3d-unet-kits19/run_eval.sh

.PHONY: bert
bert:
	-bash language/bert/run_eval.sh

.PHONY: gpt-j
gpt-j:
	-bash language/gpt-j/run_eval.sh

# verified evaluation log
.PHONY: log_all
log_all: log_resnet log_retinanet log_3d-unet log_bert log_rnnt log_gpt-j

.PHONY: log_resnet
log_resnet:
	-dvc pull logs/internal/resnet.dvc

.PHONY: log_retinanet
log_retinanet:
	-dvc pull logs/internal/retinanet.dvc

.PHONY: log_3d-unet
log_3d-unet:
	-dvc pull logs/internal/3d-unet.dvc

.PHONY: log_bert
log_bert:
	-dvc pull logs/internal/bert-squad.dvc

.PHONY: log_gpt-j
log_gpt-j:
	-dvc pull logs/internal/gpt-j.dvc

.PHONY: log_rnnt
log_rnnt:
	-dvc pull logs/internal/rnnt.dvc
