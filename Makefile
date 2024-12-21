# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := podcast2

# PAPER (updated) dec 2022
# top1
#PKL_IDENTIFIER := full-incorrect-PAPER2-hs
PKL_IDENTIFIER := full-correct-PAPER2-hs
# OG
#PKL_IDENTIFIER := full-paper-gpt2-xlhs
#top5
#PKL_IDENTIFIER := full-top5-correct-PAPER2-hs 
#PKL_IDENTIFIER := full-top5-incorrect-PAPER2-hs

# podcast electeode IDs
E_LIST :=  $(shell seq 1 115)
SID := 777

# number of permutations (goes with SH and PSH)
NPERM := 1

# Choose the lags to run for.
LAGS := {-2000..2000..25}

CONVERSATION_IDX := 0

# Choose which set of embeddings to use
EMB := gpt2-xl

CNXT_LEN := 1024

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with
ALIGN_WITH := gpt2-xl
ALIGN_TGT_CNXT_LEN := 1024

# Specify the minimum word frequency
MWF := 1

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
#SH := --shuffle
#PSH := --phase-shuffle


# Choose whether to PCA the embeddings before regressing or not
PCA := --pca-flag
PCA_TO := 50

# num layers
NUM_LAYERS := 48
LAYERS := {1..48..1}
#LAYERS :=9
OUT_NAME := 'reproduction-correct-aligned'
# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
#CMD := sbatch submit1.sh
CMD := bash submit1.sh
#CMD := python


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

run-layered-sig-encoding:
	mkdir -p logs 
	mkdir -p results
	for layer in $(LAYERS); do \
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER)$$layer \
			--conversation-id $(CONVERSATION_IDX) \
			--sig_elec_file all160_sig_enc.csv \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--part-of-model standard \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			--layer_idx $$layer \
			--output-parent-dir $(OUT_NAME)-hs$$layer \
			--output-prefix ''; \
	done

verify-encoding: 
	bash code/verify_encoding.sh $(OUT_NAME)

plot-layered-sig-encoding:
	mkdir -p results/figures
	$(CMD) code/plot_results.py \
		--encoding_name $(OUT_NAME) \
		--num_layers $(NUM_LAYERS); \
