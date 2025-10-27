import argparse
from dataclasses import MISSING, fields
from pathlib import Path
from typing import List, Optional, Union, get_args, get_origin


def _unwrap_optional(t):
    """Return the first non-None type if t is Optional[...]; otherwise return t unchanged."""
    if get_origin(t) is Union:
        return next((a for a in get_args(t) if a is not type(None)), str)  # noqa: E721
    return t

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_config_args(config_class):
    """
    Given a config data class, parse all CLI args relevant only to it,
    create an instance of that config class,
    and update any default args passed in through CLI.
    """
    parser = argparse.ArgumentParser()
    for f in fields(config_class):
        # work out the CLI type and default
        arg_type = _unwrap_optional(f.type)

        # Compute the default value for help text
        if f.default is not MISSING:
            default_val = f.default
        elif f.default_factory is not MISSING: # for default_factory attrs
            default_val = f.default_factory()
        else:
            default_val = None

        # Paths arrive as strings on the CLI
        if arg_type is Path:
            arg_type_cli = str
        else:
            arg_type_cli = arg_type

        # add CLI flags
        # bools -> nice on/off behaviour
        if arg_type is bool:
            parser.add_argument(
                f"--{f.name}",
                type=_str2bool,
                default=None,
                nargs="?",
                const=True,
                help=f"override {config_class.__name__}.{f.name} (default: {default_val})",
            )

        # lists -> space‑separated values
        elif get_origin(arg_type) in (list, List):
            inner = get_args(arg_type)[0] if get_args(arg_type) else str
            if inner is Path:
                inner = str
            parser.add_argument(
                f"--{f.name}",
                nargs="+",
                type=inner,
                default=None,
                help=f"override {config_class.__name__}.{f.name} (default: {default_val})",
            )

        # everything else
        else:
            parser.add_argument(
                f"--{f.name}",
                type=arg_type_cli,
                default=None,
                help=f"override {config_class.__name__}.{f.name} (default: {default_val})",
            )

    args, _ = parser.parse_known_args()

    config = config_class()
    for name, val in vars(args).items():
        if val is not None:
            # cast CLI strings back to Path objects where needed
            field_obj = next(f for f in fields(config_class) if f.name == name)
            target_type = _unwrap_optional(field_obj.type)

            if target_type is Path:
                if isinstance(val, list):
                    val = [Path(v) for v in val]
                else:
                    val = Path(val)

            setattr(config, name, val)

    return config, args