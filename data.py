"""
Dataset downloader for Sentence-BERT paper replication.

Directory structure
-------------------

data/
│
├── NLI/
│   ├── SNLI/
│   ├── MultiNLI/
│   └── NLI/
│
├── STS/
│   ├── STS12/
│   ├── STS13/
│   ├── STS14/
│   ├── STS15/
│   ├── STS16/
│   ├── STSb/
│   └── SICK-R/
│
└── SentEval/
    ├── MR/
    ├── CR/
    ├── SUBJ/
    ├── MPQA/
    ├── SST/
    ├── TREC/
    └── MRPC/


Datasets
--------

NLI:
    SNLI
    MultiNLI
    Combined NLI

STS:
    STS12
    STS13
    STS14
    STS15
    STS16
    STS Benchmark
    SICK Relatedness

SentEval:
    MR
    CR
    SUBJ
    MPQA
    SST
    TREC
    MRPC
    
Original source splits are preserved where they exist.
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path
import urllib.request
import zipfile
import tarfile
import shutil
import tempfile
from datasets import (load_dataset,load_from_disk,concatenate_datasets,Dataset,DatasetDict)

# DIRECTORIES

DATA_DIR = Path("data")
NLI_DIR  = DATA_DIR / "NLI"
STS_DIR  = DATA_DIR / "STS"
SENTEVAL_DIR = DATA_DIR / "SentEval"

NLI_DIR.mkdir(parents=True,exist_ok=True)
STS_DIR.mkdir(parents=True,exist_ok=True)
SENTEVAL_DIR.mkdir(parents=True,exist_ok=True)

# GENERAL HUGGING FACE DATASET LOADER

def download_hf_dataset(dataset_name,save_path,config=None,trust_remote_code=False):
    """
    Download a Hugging Face dataset and save it locally.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset ID.

    save_path : Path
        Local directory where the dataset will be saved.

    config : str or None
        Optional dataset configuration.

    trust_remote_code : bool
        Required by some older Hugging Face datasets such as TREC.
    """

    save_path = Path(save_path)
    if save_path.exists():
        print(f"[EXISTS] {save_path}")
        return load_from_disk(str(save_path))

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print()
    print(f"Downloading: {dataset_name}")

    if config is None:
        dataset = load_dataset(dataset_name,trust_remote_code=trust_remote_code)
    else:
        dataset = load_dataset(dataset_name,config,trust_remote_code=trust_remote_code)

    print(f"Saving to: {save_path}")
    dataset.save_to_disk(str(save_path))
    print(f"[SAVED] {save_path}")
    return dataset


# ============================================================
# 1. NLI DATASETS
# ============================================================

def download_nli():
    """
    Download:

        SNLI
        MultiNLI

    Then create:

        NLI = SNLI train + MultiNLI train

    Invalid labels (-1) are removed only from the combined
    training dataset.
    """

    print()
    print("=" * 70)
    print("NLI DATASETS")
    print("=" * 70)

    # SNLI
    snli = download_hf_dataset(dataset_name="stanfordnlp/snli",save_path=NLI_DIR / "SNLI")
    # MultiNLI
    mnli = download_hf_dataset(dataset_name="nyu-mll/multi_nli",save_path=NLI_DIR / "MultiNLI")
    # Combined NLI
    combined_path = NLI_DIR / "NLI"
    if combined_path.exists():

        print("\n[EXISTS] Combined NLI")
        return load_from_disk(str(combined_path))

    print()
    print("Creating combined NLI training dataset...")
    # Only training portions are used for the SBERT NLI
    # training described in the paper.
    snli_train = snli["train"]
    mnli_train = mnli["train"]

    # Keep common columns.
    columns = ["premise","hypothesis","label"]
    snli_train = snli_train.select_columns(columns)
    mnli_train = mnli_train.select_columns(columns)

    # Remove invalid / unlabeled examples.
    snli_train = snli_train.filter(lambda x: x["label"] in [0, 1, 2])
    mnli_train = mnli_train.filter(lambda x: x["label"] in [0, 1, 2])
    # Combine.
    nli = concatenate_datasets([snli_train,mnli_train])
    # Reproducible shuffle.
    nli = nli.shuffle(seed=42)
    # Save.
    nli.save_to_disk(str(combined_path))
    print(f"[SAVED] {combined_path}")
    print(f"Combined NLI pairs: {len(nli):,}")
    return nli


# ============================================================
# 2. STS DATASETS
# ============================================================

def download_sts():
    """
    Download all STS datasets used in the SBERT paper.

    STS12
    STS13
    STS14
    STS15
    STS16
    STS Benchmark
    SICK-R
    """

    print()
    print("=" * 70)
    print("STS DATASETS")
    print("=" * 70)

    sts_datasets = {

        "STS12":"mteb/sts12-sts",
        "STS13":"mteb/sts13-sts",
        "STS14":"mteb/sts14-sts",
        "STS15":"mteb/sts15-sts",
        "STS16":"mteb/sts16-sts",
        "STSb" :"mteb/stsbenchmark-sts",
        "SICK-R":"mteb/sickr-sts"}

    for folder_name, dataset_name in sts_datasets.items():
        download_hf_dataset(dataset_name=dataset_name,save_path=STS_DIR / folder_name)


# ============================================================
# 3. SENTEVAL: MR / CR / SUBJ / MPQA
# ============================================================

# This is the same SentEval archive referenced by the official SentEval download script.
SENTEVAL_CLASSIFICATION_URL = (
    "https://dl.fbaipublicfiles.com/"
    "senteval/senteval_data/"
    "datasmall_NB_ACL12.zip"
)


def download_file(url,destination):
    """
    Download a file using urllib.
    """

    print()
    print(f"Downloading:")
    print(url)
    urllib.request.urlretrieve(url,destination)
    print(f"Downloaded:")
    print(destination)


def create_binary_dataset(positive_file,negative_file):
    """
    Create a Hugging Face Dataset from positive and negative
    SentEval text files.

    label:
        0 = negative
        1 = positive
    """

    sentences = []
    labels = []

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    with open(negative_file,"r",encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(0)

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    with open(positive_file,"r",encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(1)

    return Dataset.from_dict({"sentence": sentences,"label": labels})


def download_senteval_binary_datasets():
    """
    Download the official SentEval binary classification archive
    containing:

        MR
        CR
        SUBJ
        MPQA
    """

    print()
    print("=" * 70)
    print("SENTEVAL: MR / CR / SUBJ / MPQA")
    print("=" * 70)

    # --------------------------------------------------------
    # Temporary working directory
    # --------------------------------------------------------

    temp_dir = DATA_DIR / "_senteval_temp"
    temp_dir.mkdir(parents=True,exist_ok=True)
    zip_path = temp_dir / "data_classif.zip"
    extracted_dir = temp_dir / "data_bin_classif"

    # --------------------------------------------------------
    # Download archive
    # --------------------------------------------------------

    if not zip_path.exists():

        download_file(SENTEVAL_CLASSIFICATION_URL,zip_path)
    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    if not extracted_dir.exists():

        print("\nExtracting SentEval classification data...")
        extracted_dir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(extracted_dir)

    # --------------------------------------------------------
    # Locate actual data directory
    # --------------------------------------------------------

    root = extracted_dir / "data"

    # --------------------------------------------------------
    # MR
    # --------------------------------------------------------

    mr_path = SENTEVAL_DIR / "MR"

    if not mr_path.exists():

        print("\nCreating MR...")
        mr_source = root / "rt10662"

        dataset = create_binary_dataset(
            positive_file=(mr_source / "rt-polarity.pos"),
            negative_file=(mr_source / "rt-polarity.neg"))

        dataset.save_to_disk(str(mr_path))
        print(f"[SAVED] {mr_path}")
    else:
        print(f"[EXISTS] {mr_path}")

    # --------------------------------------------------------
    # CR
    # --------------------------------------------------------

    cr_path = SENTEVAL_DIR / "CR"

    if not cr_path.exists():
        print("\nCreating CR...")
        cr_source = root / "customerr"
        dataset = create_binary_dataset(
            positive_file=(cr_source / "custrev.pos"),
            negative_file=(cr_source / "custrev.neg"))
        dataset.save_to_disk(str(cr_path))
        print(f"[SAVED] {cr_path}")
    else:
        print(f"[EXISTS] {cr_path}")

    # --------------------------------------------------------
    # SUBJ
    # --------------------------------------------------------

    subj_path = SENTEVAL_DIR / "SUBJ"
    if not subj_path.exists():
        print("\nCreating SUBJ...")
        subj_source = root / "subj"
        sentences = []
        labels = []

        # Subjective = 1
        with open(subj_source / "subj.subjective","r",encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
                    labels.append(1)

        # Objective = 0
        with open(subj_source / "subj.objective","r",encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
                    labels.append(0)

        dataset = Dataset.from_dict({"sentence": sentences,"label": labels})
        dataset.save_to_disk(str(subj_path))
        print(f"[SAVED] {subj_path}")
    else:
        print(f"[EXISTS] {subj_path}")

    # --------------------------------------------------------
    # MPQA
    # --------------------------------------------------------

    mpqa_path = SENTEVAL_DIR / "MPQA"

    if not mpqa_path.exists():

        print("\nCreating MPQA...")

        mpqa_source = root / "mpqa"
        dataset = create_binary_dataset(positive_file=(mpqa_source / "mpqa.pos"),
                                        negative_file=(mpqa_source / "mpqa.neg"))
        dataset.save_to_disk(str(mpqa_path))
        print(f"[SAVED] {mpqa_path}")

    else:
        print(f"[EXISTS] {mpqa_path}")

    # Remove temporary files
    try:
        shutil.rmtree(temp_dir)
        print("\nTemporary SentEval files removed.")
    except Exception:
        print("\nCould not remove temporary directory:")
        print(temp_dir)


# ============================================================
# 4. SENTEVAL: SST
# ============================================================

def download_sst():
    """
    Download Stanford Sentiment Treebank.

    This is the SST binary sentiment dataset used by
    SentEval.
    """

    print()
    print("=" * 70)
    print("SENTEVAL: SST")
    print("=" * 70)

    download_hf_dataset(
        dataset_name="stanfordnlp/sst2",
        save_path=SENTEVAL_DIR / "SST"
    )


# ============================================================
# 5. SENTEVAL: TREC
# ============================================================

def download_trec():
    """
    Download TREC question classification.

    The Hugging Face TREC dataset contains:
        train
        test

    We do not create or modify these splits.
    """

    print()
    print("=" * 70)
    print("SENTEVAL: TREC")
    print("=" * 70)

    download_hf_dataset(
        dataset_name="CogComp/trec",
        save_path=SENTEVAL_DIR / "TREC",
        trust_remote_code=True
    )


# ============================================================
# 6. SENTEVAL: MRPC
# ============================================================

def download_mrpc():
    """
    Download Microsoft Research Paraphrase Corpus.

    The original SentEval downloader uses the same MRPC
    train/test files. The Hugging Face GLUE mirror provides
    the dataset in its original train/validation/test form.
    """

    print()
    print("=" * 70)
    print("SENTEVAL: MRPC")
    print("=" * 70)

    download_hf_dataset(
        dataset_name="nyu-mll/glue",
        config="mrpc",
        save_path=SENTEVAL_DIR / "MRPC"
    )


# ============================================================
# 7. ALL SENTEVAL DATASETS
# ============================================================

def download_senteval():

    print()
    print("=" * 70)
    print("SENTEVAL DATASETS")
    print("=" * 70)

    # --------------------------------------------------------
    # MR / CR / SUBJ / MPQA
    # --------------------------------------------------------

    download_senteval_binary_datasets()

    # --------------------------------------------------------
    # SST
    # --------------------------------------------------------

    download_sst()

    # --------------------------------------------------------
    # TREC
    # --------------------------------------------------------

    download_trec()

    # --------------------------------------------------------
    # MRPC
    # --------------------------------------------------------

    download_mrpc()

def main():
    print()
    print("=" * 70)
    print("SBERT PAPER DATASET DOWNLOAD")
    print("=" * 70)
    print()
    print("Data directory:")
    print(DATA_DIR.resolve())

    # NLI
    download_nli()
    # STS
    download_sts()
    # SentEval
    download_senteval()

    print()
    print("=" * 70)
    print("DATASET DOWNLOAD COMPLETED")
    print("=" * 70)

    print()
    print("Final directory:")
    print(DATA_DIR.resolve())

if __name__ == "__main__":
    main()