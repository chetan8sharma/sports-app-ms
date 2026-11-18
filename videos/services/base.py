import os
import tempfile
import hashlib


class BaseVideoService:

    def process(self, video_file, filename):
        suffix = os.path.splitext(filename)[-1] or '.mp4'
        video_bytes = video_file.read()

        # Verify integrity — temp file must be identical to upload
        upload_md5 = hashlib.md5(video_bytes).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Verify written file matches upload exactly
        with open(tmp_path, 'rb') as f:
            written_md5 = hashlib.md5(f.read()).hexdigest()

        if upload_md5 != written_md5:
            os.remove(tmp_path)
            raise ValueError("File integrity check failed during upload. Please retry.")

        try:
            return self._run_analysis(tmp_path, filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _run_analysis(self, video_path, filename):
        raise NotImplementedError("Each service must implement _run_analysis()")