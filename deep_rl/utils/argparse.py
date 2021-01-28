import inspect
try:
    from typing import Literal
except(ImportError):
    # Literal polyfill
    class _Literal:
        @classmethod
        def __getitem__(cls, key):
            tp = key[0] if isinstance(key, tuple) else key
            return type(tp)
    Literal = _Literal()


def get_parameters(function_or_cls):
    if inspect.isclass(function_or_cls):
        def collect_parameters(cls):
            collected_params = set()
            parameters = inspect.signature(cls.__init__).parameters
            has_parent = False
            for p in parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    has_parent = True
                if p.kind == inspect.Parameter.KEYWORD_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    collected_params.add(p.name)
                    yield p
            if has_parent:
                for base in cls.__bases__:
                    for p in collect_parameters(base):
                        if p.name not in collected_params:
                            yield p

        params = list(collect_parameters(function_or_cls))
    else:
        params = []
        parameters = inspect.signature(function_or_cls).parameters
        for p in parameters.values():
            if p.kind == inspect.Parameter.KEYWORD_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                params.append(p)
    return params


def is_type_compatible(annotation, type2):
    if annotation == type2:
        return True
    meta_name = getattr(getattr(annotation, '__origin__', None), '_name', None)
    if meta_name == 'Literal':
        arg_type = type(annotation.__args__[0])
        return arg_type == type2
    if meta_name == 'Union':
        return any(is_type_compatible(x, type2) for x in annotation.__args__)
    return False


def add_arguments(parser, function_or_cls, defaults=None, **kwargs):
    if defaults is None:
        defaults = dict()
    params = get_parameters(function_or_cls)
    bindables = set()
    for p in params:
        bindables.add(p.name)
        default = p.default
        override_default = False
        if p.name in kwargs:
            continue
        if p.name in defaults:
            default = defaults[p.name]
            override_default = True
        # if default is None:
        #     continue
        if p.annotation is None:
            continue
        name = p.name.replace('_', '-')
        annotation = p.annotation
        help = ''
        # Note, we will check if there already exists an action with the same type
        arg_type = None
        choices = None

        # Handle meta types
        meta_name = getattr(getattr(annotation, '__origin__', None), '_name', None)
        if meta_name == 'Literal':
            arg_type = type(annotation.__args__[0])
            choices = annotation.__args__
        elif meta_name == 'Union':
            tp = set((x for x in annotation.__args__ if not isinstance(None, x)))
            if str in tp:
                arg_type = str
            if int in tp:
                arg_type = int
            if float in tp:
                arg_type = float
        elif isinstance(default, bool):
            arg_type = bool
        elif annotation in [int, float, str, bool]:
            arg_type = annotation
        if arg_type is None:
            continue

        for existing_action in parser._actions:
            if existing_action.dest == p.name:
                break
        else:
            existing_action = None
        if existing_action is not None:
            # We will update default
            if default is not inspect._empty:
                if existing_action.default is not inspect._empty and not override_default and existing_action.default != default:
                    raise Exception(f'There are conflicting values for {p.name}, [{existing_action.default}, {p.default}]')
                parser.set_defaults(**{p.name: default})
                existing_action.help = f'{help} [{default}]'
            if not is_type_compatible(annotation, existing_action.type):
                raise Exception(f'There are conflicting types for argument {p.name}, [{arg_type}, {existing_action.type}]')
            if choices is not None:
                # Update choices in the literal
                existing_action.choices = sorted(set(existing_action.choices or {}).intersection(choices))
        else:
            if arg_type == bool:
                parser.set_defaults(**{p.name: default})
                parser.add_argument(f'--{name}', dest=p.name, action='store_true', help=f'{help} [{default}]')
                parser.add_argument(f'--no-{name}', dest=p.name, action='store_false', help=f'{help} [{default}]')
            else:
                parser.add_argument(f'--{name}', type=arg_type, choices=choices, default=default, help=f'{help} [{default}]')

    bindables.intersection_update((a.dest for a in parser._actions))
    defaults = {k: v for k, v in defaults.items() if k not in bindables}

    def bind_arguments(args):
        ret = {}
        for b in bindables:
            if hasattr(args, b):
                ret[b] = getattr(args, b)
        ret.update(**defaults)
        ret.update(**kwargs)
        return ret
    return parser, bind_arguments


def bind_arguments(args, function_or_cls):
    kwargs = {}
    args_dict = args.__dict__
    parameters = get_parameters(function_or_cls)
    for p in parameters:
        if p.name in args_dict:
            kwargs[p.name] = args_dict[p.name]
    return kwargs
