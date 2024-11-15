class Result:
    def __init__(self, result):
        self.result = result

    def __repr__(self):
        return f"Result({self.result})"
    
    def get_result(self):
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