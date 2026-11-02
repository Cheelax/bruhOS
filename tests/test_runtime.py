import threading
import time

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.trace import Sink, SinkDispatcher


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_put_red_in_box():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Put the red one in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
    assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_fall_mid_skill_recovers_and_resumes():
    body = SimBody(falls_at={4})
    ep = Runtime(body, RulesPlanner()).run("Put the red one in the box.")
    assert ep.falls == 1 and ep.recoveries == 1
    kinds = [s.kind for s in ep.steps]
    assert "recovery" in kinds
    resumed = ep.steps[kinds.index("recovery") + 1]
    assert resumed.call == ep.steps[kinds.index("recovery") - 1].call
    assert ep.success and "in_box:red_block" in body.world.events


def test_unregistered_skill_from_brain_never_reaches_body():
    body = SimBody()
    ep = Runtime(body, FixedPlanner([SkillCall("wave"), SkillCall("moonwalk")])).run("Dance.")
    assert not ep.success
    assert ep.steps[0].kind == "rejected"
    assert "wave" not in body.world.events


def test_stop_preempts_running_skill():
    class SlowBody(SimBody):
        def act(self, primitive):
            time.sleep(0.01)
            return super().act(primitive)

    body = SlowBody(World(objects=[]))          # nothing to find: detect scans until stopped
    rt = Runtime(body, FixedPlanner([SkillCall("detect", {"target": "red_block"})]))
    threading.Timer(0.05, rt.stop).start()
    t0 = time.monotonic()
    ep = rt.run("Find the red block.")
    assert time.monotonic() - t0 < 1.0
    assert ep.end_reason == "stopped" or ep.steps[-1].outcome["status"] in {"stopped", "failed"}
    assert body._halted.is_set()


def test_slow_sink_does_not_block_control_loop():
    class SlowSink(Sink):
        name = "slow"

        def emit(self, ep):
            time.sleep(2.0)
            return {}

    d = SinkDispatcher([SlowSink()])
    t0 = time.monotonic()
    ep = Runtime(SimBody(), RulesPlanner(), d).run("Wave hello.")
    assert ep.success
    assert time.monotonic() - t0 < 0.5


def test_failed_sink_does_not_crash():
    class BadSink(Sink):
        name = "bad"

        def emit(self, ep):
            raise RuntimeError("chain down")

    d = SinkDispatcher([BadSink()])
    ep = Runtime(SimBody(), RulesPlanner(), d).run("Wave hello.")
    assert d.flush(2)
    assert ep.success and d.receipts[0][2].startswith("error")


def test_out_of_scope_instruction_has_no_plan():
    ep = Runtime(SimBody(), RulesPlanner()).run("Do a backflip.")
    assert not ep.success and ep.end_reason == "no plan for instruction"


class StuckBody(SimBody):
    """A body whose stand-up action groups never get it back on its feet."""

    def _stand_up(self, front):
        self.world.events.append("stand_up_attempt")
        return False


class CountingPlanner(Planner):
    name = "counting"

    def __init__(self):
        self.plans = 0

    def plan(self, instruction, obs, catalog):
        self.plans += 1
        return [SkillCall("wave")]


def test_recovery_budget_exhausted_when_starting_fallen():
    body = StuckBody(World(objects=[], fallen=True, fell_forward=False))
    brain = CountingPlanner()
    ep = Runtime(body, brain, config=RuntimeConfig(max_recoveries=3)).run("Wave hello.")
    assert not ep.success
    assert ep.end_reason == "fallen, recovery budget exhausted"
    assert ep.recoveries == 3
    assert [(s.kind, s.call["skill"], s.outcome["status"]) for s in ep.steps] == [("recovery", "stand_up", "failed")] * 3
    assert ep.plan == [] and brain.plans == 0            # never planned while still on the floor
    assert body.world.fallen and "wave" not in body.world.events


def test_recovery_budget_exhausted_after_a_fall_mid_episode():
    body = StuckBody(falls_at={4})
    ep = Runtime(body, RulesPlanner(), config=RuntimeConfig(max_recoveries=2)).run("Put the red one in the box.")
    assert not ep.success
    assert ep.end_reason == "fallen, recovery budget exhausted"
    assert ep.falls == 1 and ep.recoveries == 2
    kinds = [s.kind for s in ep.steps]
    assert kinds.count("recovery") == 2 and kinds[-1] == "recovery"
    assert ep.steps[kinds.index("recovery") - 1].outcome["status"] == "fallen"
    assert "in_box:red_block" not in body.world.events


class ReplanningPlanner(Planner):
    """First plan fails on a target that is not in the world; ``replan`` decides what happens next."""

    name = "replanning"

    def __init__(self, second):
        self.second = second
        self.histories = []

    def plan(self, instruction, obs, catalog):
        return [SkillCall("walk_toward", {"target": "purple_block"}), SkillCall("wave")]

    def replan(self, instruction, obs, catalog, history):
        self.histories.append(history)
        return list(self.second)


def test_failed_skill_triggers_one_replan_with_history():
    body = SimBody()
    brain = ReplanningPlanner(second=[SkillCall("wave")])
    ep = Runtime(body, brain).run("Fetch the purple block.")
    assert ep.success and ep.end_reason == "completed"
    assert len(brain.histories) == 1
    failed = brain.histories[0][-1]
    assert failed["call"] == {"skill": "walk_toward", "args": {"target": "purple_block", "stop_distance_m": 0.1}}
    assert failed["outcome"]["status"] == "failed"
    assert [(s.call["skill"], s.outcome["status"]) for s in ep.steps] == [("walk_toward", "failed"), ("wave", "ok")]
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "wave", "wave"]
    assert body.world.events == ["wave"]                 # the first plan's wave never ran


def test_second_failure_ends_the_episode_without_another_replan():
    brain = ReplanningPlanner(second=[SkillCall("walk_toward", {"target": "purple_block"})])
    ep = Runtime(SimBody(), brain).run("Fetch the purple block.")
    assert not ep.success
    assert ep.end_reason.startswith("walk_toward failed: purple_block not found")
    assert len(brain.histories) == 1
    assert [s.outcome["status"] for s in ep.steps] == ["failed", "failed"]


def test_empty_replan_ends_with_the_failed_skill():
    brain = ReplanningPlanner(second=[])
    ep = Runtime(SimBody(), brain).run("Fetch the purple block.")
    assert not ep.success
    assert ep.end_reason.startswith("walk_toward failed")
    assert len(brain.histories) == 1 and len(ep.steps) == 1


def test_no_replan_when_budget_is_zero():
    brain = ReplanningPlanner(second=[SkillCall("wave")])
    ep = Runtime(SimBody(), brain, config=RuntimeConfig(max_replans=0)).run("Fetch the purple block.")
    assert not ep.success and ep.end_reason.startswith("walk_toward failed")
    assert brain.histories == [] and len(ep.steps) == 1
