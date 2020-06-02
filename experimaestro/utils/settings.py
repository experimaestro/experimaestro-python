from pathlib import Path
import marshmallow as mm

class JsonSettings():   
    @classmethod
    def load(cls, path: Path):
        if path.is_file():
            settings = cls.SCHEMA().loads(path.read_text())
        else:
            settings = cls()
        settings.path = path
        return settings

    def save(self):
        """Save the preferences"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.SCHEMA().dumps(self))


class PathField(mm.fields.Field):
    """Field that serializes to a title case string and deserializes
    to a lower case string.
    """

    def _serialize(self, value, attr, obj, **kwargs):
        return Path(value)

    def _deserialize(self, value, attr, data, **kwargs):
        return str(value.absolute())
