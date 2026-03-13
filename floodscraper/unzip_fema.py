from pathlib import Path
import zipfile
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = BASE_DIR / "data" / "fema_downloads"
UNZIPPED_DIR = BASE_DIR / "data" / "fema_unzipped"

# Change to True if you want to overwrite existing extracted folders
OVERWRITE_EXISTING = False


def safe_extract_zip(zip_path: Path, dest_folder: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_folder)


def main() -> None:
    UNZIPPED_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(DOWNLOADS_DIR.rglob("*.zip"))

    if not zip_files:
        print(f"No zip files found in: {DOWNLOADS_DIR}")
        return

    print(f"Found {len(zip_files)} zip files.\n")

    extracted_count = 0
    skipped_count = 0
    failed_count = 0

    for zip_path in zip_files:
        folder_name = zip_path.stem
        dest_folder = UNZIPPED_DIR / folder_name

        try:
            if dest_folder.exists():
                if OVERWRITE_EXISTING:
                    shutil.rmtree(dest_folder)
                else:
                    print(f"Skipping already extracted: {zip_path.name}")
                    skipped_count += 1
                    continue

            dest_folder.mkdir(parents=True, exist_ok=True)
            safe_extract_zip(zip_path, dest_folder)

            print(f"Extracted: {zip_path.name} -> {dest_folder}")
            extracted_count += 1

        except Exception as e:
            print(f"FAILED: {zip_path.name} -> {e}")
            failed_count += 1

    print("\nDone.")
    print(f"Extracted: {extracted_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Failed:    {failed_count}")


if __name__ == "__main__":
    main()