import torch

from craft_simulator import SKI, SKRI, ActionEnum, ActionsCount, SKI_length, action_registry_tensor, Simulator, action_registry, CraftResultEnum
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

action_registry_tensor = action_registry_tensor.to(device)

action2buff_matrix = (action_registry_tensor[:, SKRI.WASTE_NOT:SKRI.OBSERVE_USED + 1, SKI.CONST_1] > 0).to(torch.float32)
action2cp_matrix = action_registry_tensor[:, SKRI.CP_DELTA, SKI.CONST_1]

synthesis_actions = [
    ActionEnum.prudent_synthesis,
    ActionEnum.groundwork,
    ActionEnum.basic_synthesis,
    ActionEnum.prudent_synthesis,
    ActionEnum.careful_synthesis,
    ActionEnum.focused_synthesis,
    ActionEnum.delicate_synthesis,

    ActionEnum.veneration,
]

touch_actions = [
    ActionEnum.basic_touch,
    ActionEnum.standard_touch,
    ActionEnum.byregot_s_blessing,
    ActionEnum.prudent_touch,
    ActionEnum.focused_touch,
    ActionEnum.preparatory_touch,
    ActionEnum.delicate_synthesis,
    ActionEnum.advanced_touch,
    ActionEnum.trained_finesse,

    ActionEnum.innovation,
    ActionEnum.great_strides,

]

action2craft_optimizer_name = {"basic_synthesis": "basicSynth2",
                               "observe": "observe",
                               "basic_touch": "basicTouch",
                               "masters_mend": "mastersMend",
                               "waste_not": "wasteNot",
                               "veneration": "veneration",
                               "standard_touch": "standardTouch",
                               "great_strides": "greatStrides",
                               "innovation": "innovation",
                               "waste_not_ii": "wasteNot2",
                               "byregot_s_blessing": "byregotsBlessing",
                               "muscle_memory": "muscleMemory",
                               "careful_synthesis": "carefulSynthesis2",
                               "manipulation": "manipulation",
                               "prudent_touch": "prudentTouch",
                               "focused_synthesis": "focusedSynthesis",
                               "focused_touch": "focusedTouch",
                               "reflect": "reflect",
                               "preparatory_touch": "preparatoryTouch",
                               "groundwork": "groundwork2",
                               "delicate_synthesis": "delicateSynthesis",
                               "advanced_touch": "advancedTouch",
                               "prudent_synthesis": "prudentSynthesis",
                               "trained_finesse": "trainedFinesse", }


def generate_answer_space(input_matrix: torch.Tensor, action_queue: torch.Tensor):
    # possible_space = torch.zeros((input_matrix.size(0), ActionsCount))  # (n_batch, ActionsCount)
    forbidden_space = torch.zeros((input_matrix.size(0), ActionsCount), device=device)  # (n_batch, ActionsCount)

    # ----- criteria based on input matrix -----

    # make repeated buff action forbidden
    input_buff_matrix = (input_matrix[:, SKI.WASTE_NOT:SKI.OBSERVE_USED + 1] > 1).to(torch.float32)  # (n_batch, BuffCount)
    forbidden_buff_action_matrix = torch.einsum('ij,kj->ki', action2buff_matrix, input_buff_matrix) > 0  # (n_batch, ActionsCount)
    forbidden_space += forbidden_buff_action_matrix.to(torch.float32)

    # make cp over-usage forbidden
    forbidden_space += (input_matrix[:, SKI.CP].unsqueeze(1) + action2cp_matrix.unsqueeze(0)) < 0

    # the step 0 only muscle memory is allowed
    forbidden_space[:, [i for i in ActionEnum if i != ActionEnum.muscle_memory]] += (input_matrix[:, SKI.STEP] == 0).unsqueeze(1)

    # when step >= 1, reflect and muscle memory are not allowed
    forbidden_space[:, [i for i in ActionEnum if i in [ActionEnum.reflect, ActionEnum.muscle_memory]]] += (input_matrix[:, SKI.STEP] >= 1).unsqueeze(1)

    # the step 1 only manipulation is allowed
    forbidden_space[:, [i for i in ActionEnum if i != ActionEnum.manipulation]] += (input_matrix[:, SKI.STEP] == 1).unsqueeze(1)

    # the step 2 and 3 only waste not, waste not ii and veneration is allowed
    forbidden_space[:, [i for i in ActionEnum if i != ActionEnum.waste_not and i != ActionEnum.waste_not_ii and i != ActionEnum.veneration]] += (
            (input_matrix[:, SKI.STEP] == 2) + (input_matrix[:, SKI.STEP] == 3)
    ).unsqueeze(1)

    # when muscle memory > 0, all actions except
    # groundwork, veneration, waste not, waste not ii , manipulation
    # are not allowed
    forbidden_space[:, [ac for ac in range(ActionsCount) if ac not in [ActionEnum.groundwork, ActionEnum.veneration, ActionEnum.waste_not, ActionEnum.waste_not_ii, ActionEnum.manipulation]]] += (
            input_matrix[:, SKI.MUSCLE_MEMORY] > 0
    ).unsqueeze(1)

    # when veneration == 0, groundwork is not allowed
    forbidden_space[:, ActionEnum.groundwork] += (input_matrix[:, SKI.VENERATION] == 0)

    # when waste not > 0, prudent touch and prudent synthesis are not allowed
    # when waste not == 0, groundwork and preparatory touch are not allowed
    forbidden_space[:, [ActionEnum.prudent_touch, ActionEnum.prudent_synthesis]] += (input_matrix[:, SKI.WASTE_NOT] > 0).unsqueeze(1)
    forbidden_space[:, [ActionEnum.groundwork, ActionEnum.preparatory_touch]] += (input_matrix[:, SKI.WASTE_NOT] == 0).unsqueeze(1)

    # when observed used == 0, focused touch and focused synthesis are not allowed
    # when observed used > 0, only focused touch and focused synthesis are allowed
    forbidden_space[:, [ActionEnum.focused_touch, ActionEnum.focused_synthesis]] += (input_matrix[:, SKI.OBSERVE_USED] == 0).unsqueeze(1)
    forbidden_space[:, [ac for ac in range(ActionsCount) if ac not in [ActionEnum.focused_touch, ActionEnum.focused_synthesis]]] += (input_matrix[:, SKI.OBSERVE_USED] > 0).unsqueeze(1)

    # when standard touch used == 0, advanced touch is not allowed
    # when basic touch used == 0, standard touch is not allowed
    forbidden_space[:, ActionEnum.advanced_touch] += (input_matrix[:, SKI.STANDARD_TOUCH_USED] == 0)
    forbidden_space[:, ActionEnum.standard_touch] += (input_matrix[:, SKI.BASIC_TOUCH_USED] == 0)

    # when inner quiet < 8, greate strides is not allowed
    forbidden_space[:, ActionEnum.great_strides] += (input_matrix[:, SKI.INNER_QUIET] < 8)

    # when inner quiet > 2, synthesis actions except delicate_synthesis are not allowed
    forbidden_space[:, [ac for ac in synthesis_actions if ac != ActionEnum.delicate_synthesis]] += (input_matrix[:, SKI.INNER_QUIET] > 2).unsqueeze(1)

    # when DUR < FINAL_BASE_DUR * 2, groundwork is not allowed
    forbidden_space[:, ActionEnum.groundwork] += (input_matrix[:, SKI.DUR] < input_matrix[:, SKI.FINAL_BASE_DUR] * 2)

    # when inner quiet < 10, training finesse and byregot's blessing is not allowed
    forbidden_space[:, [ActionEnum.trained_finesse, ActionEnum.byregot_s_blessing, ActionEnum.focused_touch]] += (input_matrix[:, SKI.INNER_QUIET] < 10).unsqueeze(1)

    # when inner quiet > 5 and innovation == 0, touch actions except innovation and great strides are not allowed
    forbidden_space[:, [ac for ac in touch_actions if ac not in [ActionEnum.innovation, ActionEnum.great_strides]]] += ((input_matrix[:, SKI.INNER_QUIET] > 5) * (input_matrix[:, SKI.INNOVATION] == 0)).unsqueeze(1)

    # when MAX PROG - PROG > 2.0 * BASE PROG, touch actions except delicate_synthesis are not allowed
    forbidden_space[:, [ac for ac in touch_actions if ac != ActionEnum.delicate_synthesis]] += (input_matrix[:, SKI.MAX_PROG] - input_matrix[:, SKI.PROG] > input_matrix[:, SKI.BASE_PROG] * 2).unsqueeze(1)

    # when  MAX PROG - PROG > 2.0 * BASE PROG, focused synthesis is not allowed
    forbidden_space[:, ActionEnum.focused_synthesis] += (input_matrix[:, SKI.MAX_PROG] - input_matrix[:, SKI.PROG] > input_matrix[:, SKI.BASE_PROG] * 2)

    # when basic synthesis used times >= 2, it is no more allowed
    forbidden_space[:, ActionEnum.basic_synthesis] += (action_queue == ActionEnum.basic_synthesis).sum(1) >= 2
    # when veneration used times >= 1, it is no more allowed
    forbidden_space[:, ActionEnum.veneration] += (action_queue == ActionEnum.veneration).sum(1) >= 1
    # when waste not used times >= 2 or waste not ii used times >= 1, they are no more allowed
    forbidden_space[:, [ActionEnum.waste_not, ActionEnum.waste_not_ii]] += (((action_queue == ActionEnum.waste_not).sum(1) >= 2) + ((action_queue == ActionEnum.waste_not_ii).sum(1) >= 1)).unsqueeze(1)
    # when manipulation used times >= 2, it is no more allowed
    forbidden_space[:, ActionEnum.manipulation] += (action_queue == ActionEnum.manipulation).sum(1) >= 2
    # when delicate synthesis used times >= 2, it is no more allowed
    forbidden_space[:, ActionEnum.delicate_synthesis] += (action_queue == ActionEnum.delicate_synthesis).sum(1) >= 2

    # master's mend is not allowed
    forbidden_space[:, ActionEnum.masters_mend] += 1

    return forbidden_space == 0


sim = Simulator.reset(torch.tensor([5720, 12900, 70, 693, 252, 272], dtype=torch.float32, device=torch.device('cuda')))
succeeded_sims = []
with torch.no_grad():
    for i in range(27):
        next_space = generate_answer_space(sim.matrix, sim.action_queue_matrix)
        sim.extend_space(next_space)
        finished_indies = sim.craft_result == CraftResultEnum.SUCCESS
        if finished_indies.any():
            best_index = torch.argmax((sim.matrix[:, SKI.QUAL] * finished_indies))
            succeeded_sims.append((sim.matrix[best_index, :], sim.action_queue_matrix[best_index, :]))
        sim.drop_finished()
        sim.keep_best(100_000)

print(repr([action2craft_optimizer_name[[*action_registry.keys()][id]] for id in succeeded_sims[-1][1]]).replace("'", '"'))
print([m[SKI.QUAL].item() for m, a in succeeded_sims])
# succeeded_sims[-1][0][SKI.DUR]
