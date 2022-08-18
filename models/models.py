import argparse


class ModelsManager:
    _models = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._models[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("-m", "--model", help="Model that should be trained")
        args, _ = parser.parse_known_args()

        for model, c in cls._models.items():
            if model != args.model:
                continue
            add_args = getattr(c, "add_args", None)
            if callable(add_args):
                parser = add_args(parser)
        return parser

    @classmethod
    def tunning_scopes(cls, args):

        for model, c in cls._models.items():
            if model != args.model:
                continue
            tunning_scopes = getattr(c, "tunning_scopes", None)
            if callable(tunning_scopes):
                scopes = tunning_scopes()
        return scopes

    def list_models(self):
        return self._models

    def build_model(self, name, **kwargs):

        assert name in self._models, f"Model {name} is unknown"
        return self._models[name](**kwargs)
