import os
import sys
import time

"""
=============================================================================
MASTER PIPELINE EXECUTION SCRIPT
Project: Judicia — Legal Domain Classification System
=============================================================================
This script executes the complete end-to-end Machine Learning pipeline in sequence:
Step 1: Data Cleaning & Preprocessing (step1_data_cleaning.py)
Step 2: Stratified Dataset Splitting (step2_dataset_splitting.py)
Step 3: Label Encoding & Mapping     (step3_label_encoding.py)
Step 4: Model Training / Fine-Tuning (step4_train_model.py)
"""

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import step1_data_cleaning
import step2_dataset_splitting
import step3_label_encoding

def main():
    start_time = time.time()

    print("\n" + "=" * 80)
    print(" JUDICIA ML PIPELINE — AUTOMATED END-TO-END EXECUTION")
    print("=" * 80 + "\n")

    print("[1/3] Executing Step 1: Data Cleaning & Preprocessing...")
    step1_data_cleaning.clean_data()

    print("[2/3] Executing Step 2: Stratified Dataset Splitting (80/10/10)...")
    step2_dataset_splitting.split_data()

    print("[3/3] Executing Step 3: Label Encoding & Mapping Generation...")
    step3_label_encoding.encode_labels()

    print("=" * 80)
    print("Pipeline Data Preprocessing Finished Successfully!")
    print("To re-train the Transformer Model (InLegalBERT), run:")
    print("      python src/step4_train_model.py")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nCompleted in {elapsed} seconds.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
