import enum
from typing import Union

import torch
import torch.nn.functional


class SimulatorKeyIndex(enum.IntEnum):
    CONST_1 = enum.auto()
    # recipe
    MAX_PROG = enum.auto()
    MAX_QUAL = enum.auto()
    MAX_DUR = enum.auto()
    MAX_CP = enum.auto()
    BASE_PROG = enum.auto()
    BASE_QUAL = enum.auto()
    # state
    STEP = enum.auto()
    PROG = enum.auto()
    QUAL = enum.auto()
    DUR = enum.auto()
    CP = enum.auto()
    # buffs
    INNER_QUIET = enum.auto()
    WASTE_NOT = enum.auto()
    MANIPULATION = enum.auto()
    GREAT_STRIDES = enum.auto()
    INNOVATION = enum.auto()
    VENERATION = enum.auto()
    MUSCLE_MEMORY = enum.auto()
    BASIC_TOUCH_USED = enum.auto()
    STANDARD_TOUCH_USED = enum.auto()
    # FINAL_APPRAISAL_USED = enum.auto()
    OBSERVE_USED = enum.auto()
    # calculated
    FINAL_BASE_PROG = enum.auto()
    FINAL_BASE_QUAL = enum.auto()
    FINAL_BASE_DUR = enum.auto()
    BYREGOTS_BLESSING_FINAL_QUAL = enum.auto()


SKI = SimulatorKeyIndex


class SimulatorReturnValueKeyIndex(enum.IntEnum):
    PROG_DELTA = enum.auto()
    QUAL_DELTA = enum.auto()
    DUR_DELTA = enum.auto()
    CP_DELTA = enum.auto()
    # buffs
    INNER_QUIET_DELTA = enum.auto()
    WASTE_NOT = enum.auto()
    MANIPULATION = enum.auto()
    GREAT_STRIDES = enum.auto()
    INNOVATION = enum.auto()
    VENERATION = enum.auto()
    MUSCLE_MEMORY = enum.auto()
    BASIC_TOUCH_USED = enum.auto()
    STANDARD_TOUCH_USED = enum.auto()
    # FINAL_APPRAISAL_USED = enum.auto()
    OBSERVE_USED = enum.auto()


SKRI = SimulatorReturnValueKeyIndex

SKI_length = max(e.value for e in SKI) + 1
SKRI_length = max(e.value for e in SKRI) + 1

action_registry = {
    # prog 120 dur 10
    "basic_synthesis": [(SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 1.2), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    # observe used cp 7
    "observe": [(SKRI.OBSERVE_USED, SKI.CONST_1, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -7.0)],
    # qual 100 cp 18 dur 10, add basic touch used
    "basic_touch": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -18.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0), (SKRI.BASIC_TOUCH_USED, SKI.CONST_1, 1.0)],
    # dur delta const +30, cp88
    "masters_mend": [(SKRI.DUR_DELTA, SKI.CONST_1, 30.0), (SKRI.CP_DELTA, SKI.CONST_1, -88.0)],
    # "hasty_touch":
    # "rapid_synthesis":
    # "tricks_of_the_trade":
    # waste not used const 8, cp 56
    "waste_not": [(SKRI.WASTE_NOT, SKI.CONST_1, 4.0), (SKRI.CP_DELTA, SKI.CONST_1, -56.0)],
    # veneration used const 4, cp18
    "veneration": [(SKRI.VENERATION, SKI.CONST_1, 4.0), (SKRI.CP_DELTA, SKI.CONST_1, -18.0)],
    # standard touch used const 1, cp STANDARD_TOUCH_FINAL_CP, qual 125, dur 10
    "standard_touch": [(SKRI.STANDARD_TOUCH_USED, SKI.CONST_1, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -32.0), (SKRI.CP_DELTA, SKI.BASIC_TOUCH_USED, 14.0), (SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.25), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    # great strides used const 3, cp 32
    "great_strides": [(SKRI.GREAT_STRIDES, SKI.CONST_1, 3.0), (SKRI.CP_DELTA, SKI.CONST_1, -32.0)],
    # innovation used const 4, cp18
    "innovation": [(SKRI.INNOVATION, SKI.CONST_1, 4.0), (SKRI.CP_DELTA, SKI.CONST_1, -18.0)],
    # final appraisal used const 1, cp 1
    # "final_appraisal": [(SKRI.FINAL_APPRAISAL_USED, SKI.CONST_1, 5.0), (SKRI.CP_DELTA, SKI.CONST_1, -1.0)],
    # waste not used const 8, cp 98
    "waste_not_ii": [(SKRI.WASTE_NOT, SKI.CONST_1, 8.0), (SKRI.CP_DELTA, SKI.CONST_1, -98.0)],
    # qual BYREGOTS_BLESSING_FINAL_QUAL, cp24, dur 10, inner quiet delta -10
    "byregot_s_blessing": [(SKRI.QUAL_DELTA, SKI.BYREGOTS_BLESSING_FINAL_QUAL, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -24.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0), (SKRI.INNER_QUIET_DELTA, SKI.CONST_1, -99.0)],
    # "precise_touch":
    # muscle memory used const 5, cp 6, prog 300, dur 10 and add penalty for usage after step 0
    "muscle_memory": [(SKRI.MUSCLE_MEMORY, SKI.CONST_1, 5.0), (SKRI.CP_DELTA, SKI.CONST_1, -6.0), (SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 3.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0),
                      (SKRI.PROG_DELTA, SKI.STEP, -1e4), (SKRI.MUSCLE_MEMORY, SKI.STEP, -5),
                      ],
    # dur 10 cp 7 prog 180
    "careful_synthesis": [(SKRI.CP_DELTA, SKI.CONST_1, -7.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0), (SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 1.8)],
    # manipulation used const 8, cp 96
    "manipulation": [(SKRI.MANIPULATION, SKI.CONST_1, 8.0), (SKRI.CP_DELTA, SKI.CONST_1, -96.0)],
    # qual 100 dur 5 cp 25
    "prudent_touch": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -25.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -0.5)],
    # prog 200 dur 10 cp 5
    "focused_synthesis": [(SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 2.0), (SKRI.CP_DELTA, SKI.CONST_1, -5.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    # qual 150 dur 10 cp 18
    "focused_touch": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.5), (SKRI.CP_DELTA, SKI.CONST_1, -18.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    #  qual 100 dur 10 cp 6 inner quiet +1 and add penalty for usage after step 0
    "reflect": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -6.0), (SKRI.INNER_QUIET_DELTA, SKI.CONST_1, 1.0),
                (SKRI.QUAL_DELTA, SKI.STEP, -1e4)
                ],
    # qual 200 dur 20 cp 40 inner quiet +1
    "preparatory_touch": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 2.0), (SKRI.CP_DELTA, SKI.CONST_1, -40.0), (SKRI.INNER_QUIET_DELTA, SKI.CONST_1, 1.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -2.0)],
    # prog 360 dur 20 cp 18
    "groundwork": [(SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 3.6), (SKRI.CP_DELTA, SKI.CONST_1, -18.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -2.0)],
    # qual 100 prog 100 cp 32 dur 10
    "delicate_synthesis": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.0), (SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -32.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    # "intensive_synthesis":
    # "trained_eye":
    # qual 150  dur 10 cp ADVANCED_TOUCH_FINAL_CP
    "advanced_touch": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.5), (SKRI.CP_DELTA, SKI.CONST_1, -46.0), (SKRI.CP_DELTA, SKI.STANDARD_TOUCH_USED, 28.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -1.0)],
    # prog 180 dur 5 cp 18
    "prudent_synthesis": [(SKRI.PROG_DELTA, SKI.FINAL_BASE_PROG, 1.8), (SKRI.CP_DELTA, SKI.CONST_1, -18.0), (SKRI.DUR_DELTA, SKI.FINAL_BASE_DUR, -0.5)],
    # qual 100 cp 32 and add penalty for usage with iq < 10
    "trained_finesse": [(SKRI.QUAL_DELTA, SKI.FINAL_BASE_QUAL, 1.0), (SKRI.CP_DELTA, SKI.CONST_1, -32.0),
                        (SKRI.QUAL_DELTA, SKI.INNER_QUIET, 1e4), (SKRI.QUAL_DELTA, SKI.CONST_1, -1e5), ],
    # "careful_observation":
    # "heart_and_soul":
}


def generate_action_registry_tensor():
    action_registry_list = [action_registry[k] for k in action_registry]
    action_registry_tensor = torch.zeros((len(action_registry_list), SKRI_length, SKI_length))
    for i in range(len(action_registry_list)):
        for r, k, v in action_registry_list[i]:
            action_registry_tensor[i, r, k] = v
    return action_registry_tensor


action_registry_tensor = generate_action_registry_tensor()
ActionsCount = action_registry_tensor.shape[0]

ActionEnum = enum.IntEnum("ActionEnum", [*action_registry.keys()], start=0)


class CraftResultEnum(enum.IntEnum):
    PENDING = 0
    SUCCESS = 1
    FAILURE = 2


class Simulator:
    matrix: torch.Tensor
    craft_result: torch.Tensor
    action_queue_matrix: torch.Tensor
    action_registry_tensor: torch.Tensor

    @staticmethod
    def reset(recipe: torch.Tensor = None, count: int = 1):
        if recipe is None:
            recipe = torch.tensor([6600, 14040, 70, 702, 265, 262], dtype=torch.float32)
        recipe_matrix = recipe.unsqueeze(0).repeat(count, 1)
        return Simulator(recipe_matrix)

    def extend_space(self, possible_space: torch.Tensor):
        nonzeros = possible_space.nonzero()
        self.matrix = self.matrix[nonzeros[:, 0], :]
        self.craft_result = self.craft_result[nonzeros[:, 0]]
        self.action_queue_matrix = self.action_queue_matrix[nonzeros[:, 0], :]
        result_action_list = nonzeros[:, 1]
        self.run_action(result_action_list)

    def __init__(self, input_tensor: torch.Tensor):
        self.matrix = torch.zeros((input_tensor.size(0), SKI_length), device=input_tensor.device)
        self.matrix[:, SKI.MAX_PROG:SKI.STEP] = input_tensor
        self.matrix[:, [SKI.DUR, SKI.CP]] = self.matrix[:, [SKI.MAX_DUR, SKI.MAX_CP]]
        self.matrix[:, [SKI.CONST_1]] = 1.0
        self.craft_result = torch.zeros((input_tensor.size(0),), dtype=torch.int32, device=input_tensor.device)
        self.action_queue_matrix = torch.zeros((input_tensor.size(0), 0), dtype=torch.int32, device=input_tensor.device)
        self.action_registry_tensor = action_registry_tensor.to(input_tensor.device)

    def lerp(self, idx: Union[slice, int], min_=None, max_=None):
        if max_ is not None:
            self.matrix[:, idx] = torch.minimum(
                self.matrix[:, idx],
                torch.ones_like(self.matrix[:, idx]) * max_)
        if min_ is not None:
            self.matrix[:, idx] = torch.maximum(
                self.matrix[:, idx],
                torch.ones_like(self.matrix[:, idx]) * min_)

    def tick_before(self):
        # set FINAL_BASE_PROG
        self.matrix[:, SKI.FINAL_BASE_PROG] = self.matrix[:, SKI.BASE_PROG] * (
                1 +
                (self.matrix[:, SKI.MUSCLE_MEMORY] > 0) * 1 +
                (self.matrix[:, SKI.VENERATION] > 0) * .5
        )
        # set FINAL_BASE_QUAL
        self.matrix[:, SKI.FINAL_BASE_QUAL] = self.matrix[:, SKI.BASE_QUAL] * (
                1 +
                (self.matrix[:, SKI.GREAT_STRIDES] > 0) * 1 +
                (self.matrix[:, SKI.INNOVATION] > 0) * .5
        ) * (self.matrix[:, SKI.INNER_QUIET] * 0.1 + 1.0)
        # set FINAL_BASE_DUR
        self.matrix[:, SKI.FINAL_BASE_DUR] = 10 - 5 * (self.matrix[:, SKI.WASTE_NOT] > 0)
        # set BYREGOTS_BLESSING_FINAL_PROG
        self.matrix[:, SKI.BYREGOTS_BLESSING_FINAL_QUAL] = (
                self.matrix[:, SKI.FINAL_BASE_QUAL] *
                (self.matrix[:, SKI.INNER_QUIET] * 0.2 + 1.0)
        )

    def action(self, action_index_vector: torch.Tensor):
        action_tensor = self.action_registry_tensor[action_index_vector, :, :]
        action_result_matrix = torch.einsum('ijk,ik->ij', action_tensor, self.matrix)
        # lerp dur delta down to 5
        action_result_matrix[:, SKRI.DUR_DELTA] = torch.where(
            action_result_matrix[:, SKRI.DUR_DELTA] < 0,
            torch.minimum(
                action_result_matrix[:, SKRI.DUR_DELTA],
                torch.ones_like(action_result_matrix[:, SKRI.DUR_DELTA]) * -5
            ), action_result_matrix[:, SKRI.DUR_DELTA]
        )
        # print(self.repr_matrix(action_result_matrix))
        # remove Great strides and Muscle Memory
        self.matrix[:, SKI.GREAT_STRIDES] = (~(action_result_matrix[:, SKRI.QUAL_DELTA] > 0)) * self.matrix[:, SKI.GREAT_STRIDES]
        self.matrix[:, SKI.MUSCLE_MEMORY] = (~(action_result_matrix[:, SKRI.PROG_DELTA] > 0)) * self.matrix[:, SKI.MUSCLE_MEMORY]
        # tick manipulation
        manipulation_delta = 5 * (torch.logical_and(self.matrix[:, SKI.MANIPULATION] > 0, self.matrix[:, SKI.DUR] + action_result_matrix[:, SKRI.DUR_DELTA] > 0))
        self.matrix[:, SKI.DUR] += manipulation_delta
        self.lerp(SKI.DUR, max_=self.matrix[:, SKI.MAX_DUR] + manipulation_delta)
        # tick buff duration
        self.matrix[:, SKI.WASTE_NOT:SKI.OBSERVE_USED + 1] -= 1
        self.lerp(slice(SKI.WASTE_NOT, SKI.OBSERVE_USED + 1), min_=0)
        # step +1
        self.matrix[:, SKI.STEP] += 1
        # apply action effects
        self.matrix[:, SKI.PROG:SKI.CP + 1] += action_result_matrix[:, SKRI.PROG_DELTA:SKRI.CP_DELTA + 1]
        # update craft result
        self.craft_result[self.matrix[:, SKI.PROG] >= self.matrix[:, SKI.MAX_PROG]] = CraftResultEnum.SUCCESS
        self.craft_result[
            torch.logical_and(
                torch.logical_or(
                    self.matrix[:, SKI.DUR] <= 0,
                    self.matrix[:, SKI.STEP] >= 30
                ),
                self.craft_result != CraftResultEnum.SUCCESS
            )
        ] = CraftResultEnum.FAILURE
        return action_result_matrix

    def tick_after(self, action_result_matrix: torch.Tensor):
        # apply action buff effects
        self.matrix[:, SKI.WASTE_NOT:SKI.OBSERVE_USED + 1] = torch.where(
            action_result_matrix[:, SKRI.WASTE_NOT:SKRI.OBSERVE_USED + 1] > 0,
            action_result_matrix[:, SKRI.WASTE_NOT:SKRI.OBSERVE_USED + 1],
            self.matrix[:, SKI.WASTE_NOT:SKI.OBSERVE_USED + 1]
        )
        # tick inner quiet
        self.matrix[:, SKI.INNER_QUIET] += torch.where(
            action_result_matrix[:, SKRI.QUAL_DELTA] > 0,
            action_result_matrix[:, SKRI.INNER_QUIET_DELTA] + 1,
            0
        )
        self.lerp(SKI.INNER_QUIET, max_=10, min_=0)

        self.matrix[:, SKI.PROG:SKI.QUAL + 1] = torch.floor(self.matrix[:, SKI.PROG:SKI.QUAL + 1])

    def __repr__(self):
        s = ''
        for i in range(self.matrix.size(0)):
            s += f'Simulator {i}:\n'
            for j in range(self.matrix.size(1)):
                if self.matrix[i, j] != 0:
                    s += f'\t{[name for name, member in SKI.__members__.items() if member.value == j][0]}: {self.matrix[i, j]}\n'
        return s

    def run_action(self, action_index_vector: torch.Tensor):
        self.tick_before()
        action_result = self.action(action_index_vector)
        self.action_queue_matrix = torch.cat((self.action_queue_matrix, action_index_vector.unsqueeze(1)), dim=1)
        self.tick_after(action_result)

    def score(self):
        score = (
                        self.matrix[:, SKI.CP] +
                        self.matrix[:, SKI.PROG] / self.matrix[:, SKI.MAX_PROG] * self.matrix[:, SKI.MAX_CP] * 0.5 +
                        self.matrix[:, SKI.QUAL] / self.matrix[:, SKI.MAX_QUAL] * self.matrix[:, SKI.MAX_CP] * 0.6 +
                        self.matrix[:, SKI.DUR] / 40 * 96 +
                        self.matrix[:, SKI.INNER_QUIET] * 4 +
                        self.matrix[:, SKI.WASTE_NOT] / 3 * 50 * .6 +
                        self.matrix[:, SKI.MANIPULATION] / 8 * 96 * .6 +
                        (self.matrix[:, SKI.GREAT_STRIDES] > 0) * 32 * .6 +
                        self.matrix[:, SKI.INNOVATION] / 4 * 18 * .6 +
                        self.matrix[:, SKI.VENERATION] / 4 * 18 * .6 +
                        (self.matrix[:, SKI.MUSCLE_MEMORY] > 0) * 6 +
                        (self.matrix[:, SKI.BASIC_TOUCH_USED] > 0) * 4 +
                        (self.matrix[:, SKI.STANDARD_TOUCH_USED] > 0) * 6 +
                        (self.matrix[:, SKI.OBSERVE_USED] > 0) * 7
                ) / self.matrix[:, SKI.MAX_CP]
        return score

    def filter(self, keep_index: torch.Tensor):
        self.matrix = self.matrix[keep_index, :]
        self.craft_result = self.craft_result[keep_index]
        self.action_queue_matrix = self.action_queue_matrix[keep_index, :]

    def drop_finished(self):
        self.filter(self.craft_result == CraftResultEnum.PENDING)

    def randomly_drop(self, keep_count: int):
        self.filter(torch.randperm(self.matrix.size(0), device=self.matrix.device)[:keep_count])

    def keep_best(self, keep_count: int):
        self.filter(torch.argsort(self.score())[-keep_count:])
