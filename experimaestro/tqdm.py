from tqdm.auto import tqdm as std_tqdm
from experimaestro import progress


class xpm_tqdm(std_tqdm):
    """XPM wrapper for experimaestro that automatically reports progress to the server"""

    def __init__(self, **kwargs):
        # Disable on non TTY
        kwargs["disable"] = None

        super().__init__(**kwargs)

    def update(self, n=1):
        displayed = super().update(n)
        if displayed:
            # Get the position
            pos = abs(self.pos)
            # progress()
            print("YO SOME PROGRESS", self.format_dict)
        return displayed


tqdm = xpm_tqdm
