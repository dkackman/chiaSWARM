import os
import soundfile
import mimetypes
from diffusers.utils import export_to_video

class Result:
    def __init__(self, result, properties={}):
        self.result = result
        self.properties = properties

    def __repr__(self):
        return f"Result({self.result})"
    
    def get_raw_result(self):
        return self.result
    
    def get_output(self):
        if hasattr(self.result, "images"):
            return self.result.images[0]
        
        if hasattr(self.result, "image_embeds"):
            return self.result.image_embeds[0]

        if hasattr(self.result, "image_embeddings"):
            return self.result.image_embeddings[0]

        if hasattr(self.result, "frames"):
            return self.result.frames[0]
        
        if hasattr(self.result, "audios"):
            return self.result.audios[0].T.float().cpu().numpy()
        
        return None
    
    def guess_extension(self):
        content_type = self.properties.get("content_type", None)
        ext = mimetypes.guess_extension(content_type)
        if ext is not None:
            return ext

        if content_type == "audio/wav":
            return ".wav"
        
        return ""

    def save(self, output_dir, default_name):
        file_name = self.properties.get("file_name", default_name)  
        output_path = os.path.join(output_dir, file_name)

        output = self.get_output()
        content_type = self.properties.get("content_type", "")
        if content_type.startswith("video"):
            export_to_video(output, output_path, fps=self.properties.get("fps", 8))

        elif content_type.startswith("audio"):
            soundfile.write(output_path, output, self.properties.get("sample_rate", 44100))

        elif hasattr(output, 'save'):
            output.save(output_path)        
