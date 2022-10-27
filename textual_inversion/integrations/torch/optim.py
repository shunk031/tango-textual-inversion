from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType
from tango.integrations.torch.optim import LRScheduler

for k in TYPE_TO_SCHEDULER_FUNCTION.keys():
    LRScheduler.register(f"diffusers::{k.value}")(
        TYPE_TO_SCHEDULER_FUNCTION[SchedulerType(k.value)]
    )
