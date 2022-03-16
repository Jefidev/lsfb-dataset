import tempfile
import os
import pandas as pd

from lsfb_dataset.utils.download.dataset_downloader import DatasetDownloader


def test_csv_download_isol():

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = DatasetDownloader(tmpdir)
        ds.download_csv()

        assert os.path.exists(os.path.join(tmpdir, "clips.csv"))


def test_csv_download_cont():

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = DatasetDownloader(tmpdir, dataset="cont")
        ds.download_csv()

        assert os.path.exists(os.path.join(tmpdir, "videos.csv"))


def test_video_download_isol():

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = DatasetDownloader(tmpdir)
        csv_path = ds.download_csv()

        data = pd.read_csv(csv_path)
        row = data.iloc[0]

        ds.download_video(row)

        expected_path = os.path.join(tmpdir, row["relative_path"])

        assert os.path.exists(expected_path)
        assert os.path.getsize(expected_path) != 0


def test_video_download_cont():

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = DatasetDownloader(tmpdir, dataset="cont")
        csv_path = ds.download_csv()

        data = pd.read_csv(csv_path)
        row = data.iloc[0]

        ds.download_video(row)

        expected_path = os.path.join(tmpdir, row["relative_path"])

        assert os.path.exists(expected_path)
        assert os.path.getsize(expected_path) != 0