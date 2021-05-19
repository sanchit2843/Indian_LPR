import math


def lr_func(GLOBAL_STEPS, WARMPUP_STEPS, TOTAL_STEPS, LR_INIT, LR_END):
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (
                1
                + math.cos(
                    (GLOBAL_STEPS - WARMPUP_STEPS)
                    / (TOTAL_STEPS - WARMPUP_STEPS)
                    * math.pi
                )
            )
        )
    return float(lr)
