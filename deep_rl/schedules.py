import importlib


class Schedule:
    def __init__(self, time=None):
        self.time = time

    def __call__(self):
        return None

    def step(self, time):
        self.time = time

    def state_dict(self):
        def serialize_schedule(obj):
            if isinstance(obj, tuple):
                return tuple(map(serialize_schedule, obj))
            elif isinstance(obj, list):
                return list(map(serialize_schedule, obj))
            elif isinstance(obj, dict):
                return {k: serialize_schedule(v) for k, v in obj.items()}
            elif not isinstance(obj, Schedule):
                return obj

            state_dict = {k: serialize_schedule(v) for k, v in self.__dict__ if not k.startswith('_')}
            module = self.__class__.__module__
            state_dict['_schedule_class'] = module + '.' + self.__class__.__name__
            return state_dict
        return serialize_schedule(self)

    @classmethod
    def from_state_dict(base_cls, state_dict):
        assert '_schedule_class' in state_dict

        cls_path = state_dict.pop('_schedule_class').split('.')
        cls_path, cls_name = '.'.join(cls_path[:-1]), cls_path[-1]
        cls = getattr(importlib.import_module(cls_path), cls_name)
        import inspect
        assert base_cls in inspect.getmro(cls)
        for k, v in state_dict.items():
            if isinstance(v, dict) and '_schedule_class' in v:
                state_dict[k] = Schedule.from_state_dict(v)
        return cls(**state_dict)


class LinearSchedule(Schedule):
    def __init__(self, start, end, total_iterations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.total_iterations = total_iterations

    def __call__(self):
        return self.end + (0 if self.time >= self.total_iterations else float(self.start - self.end) * float(self.total_iterations - self.time) / float(self.total_iterations))


class ConstantSchedule(Schedule):
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def __call__(self):
        return self.value


class MultistepSchedule(Schedule):
    def _wrap_schedule(self, schedule):
        if isinstance(schedule, Schedule):
            return schedule

        return ConstantSchedule(schedule)

    def __init__(self, initial_value, steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_value = self._wrap_schedule(initial_value)
        self.steps = [(i, self._wrap_schedule(x)) for i, x in steps]

    def __call__(self):
        schedule, _ = self._get_current_schedule
        return schedule()

    def _get_current_schedule(self):
        val = self.initial_value
        time = 0
        for i, value in self.steps:
            if self.time >= i:
                val = value
                time = i
        return val, time

    def step(self, time):
        super().step(time)
        current_schedule, start_time = self._get_current_schedule()
        current_schedule.step(self.time - start_time)
