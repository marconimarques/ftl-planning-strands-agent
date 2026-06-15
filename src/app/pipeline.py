"""Pipeline — orchestrates the three models + agents for each query."""

from __future__ import annotations

import concurrent.futures
from typing import Callable, Optional

from strands import Agent

from ..domain.data_types import MILPResult, PipelineResult, ScenarioParams
from ..domain.loader import NetworkData
from ..models.lane_by_lane import run_lane_by_lane
from ..models.weighted_cycle_time import run_weighted_cycle_time
from ..agents.or_agent import run_or_agent
from ..agents.transportation_expert import run_expert_agent

# Dedicated thread for the Expert agent so it runs concurrently with LBL+WCT.
# max_workers=1 because queries are sequential — only one Expert call is ever
# in flight at a time; a single thread avoids creating unnecessary resources.
_insight_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_pipeline_from_params(
    milp_result: MILPResult,
    scenario_params: ScenarioParams,
    network: NetworkData,
    expert_agent: Agent,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Run LBL + WCT + Expert given a pre-computed MILPResult and ScenarioParams.

    When scenario_params.skip_capacity_constraints is True, re-runs the MILP solver
    without terminal/CP capacity constraints before running the remaining models.
    This is the over-capacity action-map path.
    """
    needs_solver_run = scenario_params.skip_capacity_constraints or (
        # OR Agent sometimes hallucinates infeasibility in its structured-output JSON
        # even when its internal tool call returned a valid result. Re-run to verify;
        # if genuinely infeasible the re-run confirms it. Two-level scenarios
        # (cost_a != None) are excluded — they have no single-scenario equivalent.
        not milp_result.feasible and milp_result.cost_a is None
    ) or (
        # Guard against hallucinated "feasible with 0 trucks and 0 coverage" — this
        # combination is never a valid solver result for minimize_cost / minimize_fleet
        # with any positive demand and no budget constraint set to 0.
        milp_result.feasible
        and milp_result.trucks == 0
        and milp_result.coverage_count == 0
        and milp_result.cost_a is None
        and scenario_params.budget is None
    )
    if needs_solver_run:
        if on_phase:
            on_phase("phase_milp")
        from ..models.solver import run_milp_solver
        milp_result = run_milp_solver(network, scenario_params)
        if milp_result.served_cps:
            scenario_params.served_cps = list(milp_result.served_cps)

    if milp_result.feasible and milp_result.assignments:
        scenario_params.milp_assignments = milp_result.assignments

    baseline_trucks = baseline_result.milp_result.trucks if baseline_result else None
    baseline_cost = baseline_result.milp_result.total_cost if baseline_result else None
    baseline_params = baseline_result.scenario_params if baseline_result else None
    baseline_milp = baseline_result.milp_result if baseline_result else None

    terminal_demand_totals = {
        tid: sum(network.demand.get(cp, {}).get(tid, 0.0) for cp in network.cp_ids)
        for tid in network.terminal_ids
    }

    cp_demands = {cp: sum(network.demand.get(cp, {}).values()) for cp in network.cp_ids}

    expert_future: Optional[concurrent.futures.Future[str]] = None
    if llm_insights:
        expert_future = _insight_executor.submit(
            run_expert_agent,
            expert_agent,
            milp_result,
            scenario_params,
            language,
            len(network.cp_ids),
            baseline_trucks,
            baseline_cost,
            terminal_demand_totals,
            dict(network.terminal_capacities),
            baseline_params,
            baseline_milp,
            list(network.cp_ids),
            dict(network.cp_capacities),
            cp_demands,
            network,
        )

    if on_phase:
        on_phase("phase_lbl")
    lbl_result = run_lane_by_lane(network, scenario_params)
    if on_phase:
        on_phase("phase_wct")
    wct_result = run_weighted_cycle_time(network, scenario_params)
    if on_phase:
        on_phase("phase_expert")
    insight = expert_future.result() if expert_future is not None else ""

    return PipelineResult(
        scenario_params=scenario_params,
        lbl_result=lbl_result,
        wct_result=wct_result,
        milp_result=milp_result,
        insight=insight,
        query_number=query_number,
    )


def run_pipeline(
    query: str,
    network: NetworkData,
    or_agent: Agent,
    expert_agent: Agent,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Execute the full pipeline for a user query.

    Sequence:
    1. OR Agent: interprets query, runs MILP solver → MILPResult + ScenarioParams
    2. Expert Agent submitted to background thread (LLM call, ~5-10s)
    3. Lane-by-Lane + Weighted Cycle Time: pure Python, runs while Expert is running
    4. Collect Expert result (future.result())
    5. Assemble PipelineResult
    """
    milp_result, scenario_params = run_or_agent(or_agent, query)
    return run_pipeline_from_params(
        milp_result=milp_result,
        scenario_params=scenario_params,
        network=network,
        expert_agent=expert_agent,
        query_number=query_number,
        language=language,
        llm_insights=llm_insights,
        baseline_result=baseline_result,
        on_phase=on_phase,
    )


def run_relocation_pipeline(
    network: NetworkData,
    expert_agent: Agent,
    base_params: ScenarioParams,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Run the volume redistribution scenario directly, bypassing the OR Agent.

    Inherits all operational params from base_params (speed, payload, costs, etc.)
    so the redistribution is always a fair comparison against the current baseline.
    """
    from ..models.solver import run_milp_solver

    params = ScenarioParams(
        payload=base_params.payload,
        speed_loaded=base_params.speed_loaded,
        speed_empty=base_params.speed_empty,
        availability=base_params.availability,
        overtime_hours=base_params.overtime_hours,
        overtime_cost=base_params.overtime_cost,
        variable_cost_per_km=base_params.variable_cost_per_km,
        fixed_cost_per_truck_month=base_params.fixed_cost_per_truck_month,
        working_days=base_params.working_days,
        net_driving_hours=base_params.net_driving_hours,
        terminals_active=dict(base_params.terminals_active),
        variable_cost_components=dict(base_params.variable_cost_components),
        fixed_cost_components=dict(base_params.fixed_cost_components),
        volume_redistribution=True,
        is_baseline=False,
    )

    if on_phase:
        on_phase("phase_milp")
    milp_result = run_milp_solver(network, params)
    if milp_result.feasible and milp_result.assignments:
        params.milp_assignments = milp_result.assignments

    baseline_trucks = baseline_result.milp_result.trucks if baseline_result else None
    baseline_cost = baseline_result.milp_result.total_cost if baseline_result else None
    baseline_params = baseline_result.scenario_params if baseline_result else None
    baseline_milp = baseline_result.milp_result if baseline_result else None

    terminal_demand_totals = {
        tid: sum(network.demand.get(cp, {}).get(tid, 0.0) for cp in network.cp_ids)
        for tid in network.terminal_ids
    }

    cp_demands = {cp: sum(network.demand.get(cp, {}).values()) for cp in network.cp_ids}

    expert_future: Optional[concurrent.futures.Future[str]] = None
    if llm_insights:
        expert_future = _insight_executor.submit(
            run_expert_agent,
            expert_agent,
            milp_result,
            params,
            language,
            len(network.cp_ids),
            baseline_trucks,
            baseline_cost,
            terminal_demand_totals,
            dict(network.terminal_capacities),
            baseline_params,
            baseline_milp,
            list(network.cp_ids),
            dict(network.cp_capacities),
            cp_demands,
            network,
        )

    if on_phase:
        on_phase("phase_lbl")
    lbl_result = run_lane_by_lane(network, params)
    if on_phase:
        on_phase("phase_wct")
    wct_result = run_weighted_cycle_time(network, params)
    if on_phase:
        on_phase("phase_expert")
    insight = expert_future.result() if expert_future is not None else ""

    return PipelineResult(
        scenario_params=params,
        lbl_result=lbl_result,
        wct_result=wct_result,
        milp_result=milp_result,
        insight=insight,
        query_number=query_number,
    )


def build_baseline_params(network: NetworkData) -> ScenarioParams:
    """Build the default baseline ScenarioParams from the network data."""
    return ScenarioParams(
        payload=network.payload,
        speed_loaded=network.speed_loaded,
        speed_empty=network.speed_empty,
        availability=network.availability,
        overtime_hours=network.overtime_hours,
        overtime_cost=network.overtime_cost,
        variable_cost_per_km=network.variable_cost_per_km,
        fixed_cost_per_truck_month=network.fixed_cost_per_truck_month,
        working_days=network.working_days,
        net_driving_hours=network.net_driving_hours,
        terminals_active={t: True for t in network.terminal_ids},
        min_coverage_count=None,
        budget=None,
        objective="minimize_cost",
        volume_redistribution=False,
        is_baseline=True,
        served_cps=list(network.cp_ids),
        variable_cost_components=dict(network.variable_cost_components),
        fixed_cost_components=dict(network.fixed_cost_components),
    )
