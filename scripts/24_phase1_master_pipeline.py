"""
Master script to orchestrate Phase 1: Full MPD Data Processing
Runs all data loading, co-occurrence building, and feature extraction in sequence.

Author: Adarsh Singh
Date: November 2024

Usage:
    screen -S mpd_processing
    python scripts/24_phase1_master_pipeline.py
    # Ctrl+A, D to detach
"""

import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime
import time

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'phase1_master_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """Orchestrate Phase 1 data processing pipeline."""

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.scripts_dir = self.project_dir / "scripts"
        self.scripts_dir.mkdir(exist_ok=True)

        self.pipeline_steps = [
            {
                'name': 'Load All MPD Slices',
                'script': '21_load_all_mpd_slices.py',
                'description': 'Process 1,000 MPD slice files into unified dataset',
                'estimated_time': '2-3 hours'
            },
            {
                'name': 'Build Co-occurrence Matrix',
                'script': '22_build_cooccurrence_incremental.py',
                'description': 'Build sparse co-occurrence matrix for all track pairs',
                'estimated_time': '1-2 hours'
            },
            {
                'name': 'Extract Features',
                'script': '23_extract_features_full.py',
                'description': 'Extract track and playlist features for ML',
                'estimated_time': '30-60 minutes'
            }
        ]

    def run_script(self, script_name):
        """Run a Python script and capture output."""
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        logger.info(f"Starting: {script_name}")
        start_time = time.time()

        try:
            # Run script and capture output
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.project_dir)
            )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✅ Completed: {script_name} in {elapsed_time / 60:.2f} minutes")
                return True
            else:
                logger.error(f"❌ Failed: {script_name}")
                logger.error(f"Error output: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception running {script_name}: {e}")
            return False

    def print_pipeline_overview(self):
        """Print pipeline overview."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: FULL MPD DATA PROCESSING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Project Directory: {self.project_dir}")
        logger.info(f"Total Steps: {len(self.pipeline_steps)}")
        logger.info("\nPipeline Steps:")
        for i, step in enumerate(self.pipeline_steps, 1):
            logger.info(f"\n{i}. {step['name']}")
            logger.info(f"   Script: {step['script']}")
            logger.info(f"   Description: {step['description']}")
            logger.info(f"   Estimated Time: {step['estimated_time']}")
        logger.info("\n" + "=" * 70 + "\n")

    def run_pipeline(self, start_from=0):
        """Run the complete pipeline."""

        self.print_pipeline_overview()

        total_start = time.time()
        results = []

        for i, step in enumerate(self.pipeline_steps[start_from:], start_from + 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"STEP {i}/{len(self.pipeline_steps)}: {step['name']}")
            logger.info(f"{'=' * 70}")

            success = self.run_script(step['script'])
            results.append({'step': step['name'], 'success': success})

            if not success:
                logger.error(f"\n❌ Pipeline failed at step {i}: {step['name']}")
                logger.error("Please check logs and fix errors before continuing.")
                return False

            logger.info(f"✅ Step {i} completed successfully")

        # Pipeline complete
        total_time = time.time() - total_start
        logger.info(f"\n{'=' * 70}")
        logger.info("PHASE 1 PIPELINE COMPLETED")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total execution time: {total_time / 3600:.2f} hours")
        logger.info("\nStep Results:")
        for result in results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            logger.info(f"  {result['step']}: {status}")
        logger.info(f"\n{'=' * 70}\n")

        return True

    def check_prerequisites(self):
        """Check if all required scripts are present."""
        logger.info("Checking prerequisites...")

        all_present = True
        for step in self.pipeline_steps:
            script_path = self.scripts_dir / step['script']
            if script_path.exists():
                logger.info(f"✅ Found: {step['script']}")
            else:
                logger.error(f"❌ Missing: {step['script']}")
                all_present = False

        if not all_present:
            logger.error("\nSome scripts are missing. Please ensure all scripts are in the scripts/ directory.")
            return False

        logger.info("✅ All prerequisites satisfied\n")
        return True


def main():
    """Main execution."""

    PROJECT_DIR = "/Users/drsh0755/Documents/George Washington University/Fall25/Data Mining_CSCI_6443/CSCI 6443 Data Mining - Project"

    pipeline = Phase1Pipeline(PROJECT_DIR)

    # Check prerequisites
    if not pipeline.check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        sys.exit(1)

    # Run pipeline
    logger.info("Starting Phase 1 pipeline...")
    logger.info("This will take approximately 4-6 hours to complete.")
    logger.info("Progress will be logged to the console and log files.\n")

    input("Press Enter to start the pipeline, or Ctrl+C to cancel...")

    success = pipeline.run_pipeline()

    if success:
        logger.info("\n🎉 Phase 1 Complete! Ready for Phase 2 (Experiments)")
        logger.info("Next steps:")
        logger.info("  1. Review generated data in data/processed/")
        logger.info("  2. Check statistics files")
        logger.info("  3. Proceed to Phase 2: Experiments at Scale")
    else:
        logger.error("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()