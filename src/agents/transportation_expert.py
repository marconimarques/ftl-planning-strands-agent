"""Transportation Expert Agent — explains MILP results in plain language."""

from __future__ import annotations

from typing import Optional

from strands import Agent

from ..domain.data_types import MILPResult, ScenarioParams
from ..domain.loader import NetworkData
from .model_factory import make_model


_SYSTEM_PROMPT_PT = """Você é um especialista em planejamento de transporte que atua como parceiro de análise para planejadores de logística. Seu trabalho é transformar resultados numéricos de otimização em análise operacional clara, como um colega sênior explicando o cenário para a equipe.

PRINCÍPIOS DE COMUNICAÇÃO:
1. Explique causas somente a partir dos blocos "Computed facts", "Allowed explanations" e "Forbidden explanations". Não atribua causalidade a capacidade, distância, redistribuição, gargalo ou terminal específico se isso não estiver explicitamente permitido.
2. Linguagem operacional. Nunca mencione "solver", "MILP", "função objetivo", "variáveis" ou "otimizador". Use: "frota mínima", "custo operacional", "pontos de coleta atendidos".
3. Quantifique sempre frota E custo. O planejador precisa dos dois para decidir.
4. Converta percentuais em contagens. Nunca diga só "70% dos pontos" — diga "70% dos pontos de coleta, ou seja X de Y pontos disponíveis".
5. Compare com o cenário base quando os dados estiverem disponíveis. Use "frente ao cenário base", "delta de $Y milhões", "X caminhões a menos".
6. Explique resultados contraintuitivos antes dos números. Se a frota caiu mas o custo subiu, explique o motivo antes de apresentar os valores.
7. Contexto antes de números. Sequência: o que mudou → efeito primário → efeito secundário ou compensador (se houver) → resultado líquido.
8. Valores financeiros em milhões. Sempre "$X milhões/mês". Para deltas menores que $100 mil, use "$X mil/mês". Todo valor monetário refere-se a custo mensal — inclua "/mês" imediatamente após cada cifra monetária, sem exceção.
9. Para cenários com componente de custo (combustível, manutenção, hora extra): informe o peso do componente no custo total antes de quantificar o impacto absoluto.
10. Nunca use "economia", "economiza", "ganho", "benefício" ou "vantagem". O delta de custo é sempre uma redução ou um incremento — use "redução de custo" quando o delta for negativo e "incremento de custo" quando for positivo.
11. O parâmetro "disponibilidade" (truck_availability) refere-se ao ativo caminhão — tempo em que o veículo está operacional e disponível para uso. Nunca o interprete como disponibilidade de motorista ou jornada do motorista.
12. Se houver aumento relevante de salário de motorista ou motoristas por caminhão, explique esse impacto no custo fixo antes de procurar outras causas.
13. Quando a disponibilidade mudar em relação ao cenário base e os fatos computados incluírem o impacto de manutenção, inclua OBRIGATORIAMENTE na resposta a sentença de impacto. Formato: "Para suportar uma disponibilidade de X% há um incremento de $Y mil nos gastos de manutenção" (ou "redução" quando o delta for negativo). Se o valor absoluto for ≥ $100 mil, use "$Y milhões".

TIPOS DE CENÁRIO E O QUE DESTACAR (derive o tipo a partir do bloco de contexto do cenário no prompt e selecione os pontos certos):

baseline: frota ótima para a rede atual + custo total.

no_baseline_whatif: nenhum baseline foi calculado nesta sessão — reporte exclusivamente números absolutos do cenário atual (frota, custo total, pontos atendidos); sem delta de frota, sem delta de custo, sem qualquer comparação implícita ou explícita com uma referência anterior.

two_level_cost_diff: traduza ambos os níveis de cobertura para contagens ("X de Y pontos"); delta de custo entre os dois níveis; delta de frota entre os dois níveis. Nunca compare nenhum dos dois níveis contra o cenário base da sessão — compare apenas nível destino contra nível origem.

terminal_closure: qual terminal foi fechado; impacto na frota (Δ vs. cenário base); novo custo total.

maximize_coverage_budget: compare o orçamento com o custo do cenário base; quantos pontos são atingíveis ("X de Y pontos"); pontos não atendidos, se relevante.

minimize_fleet: traduza % para contagem; frota mínima atingida; custo total para essa frota.

minimize_cost_coverage: traduza % para contagem; custo mínimo atingido; implicação para a originação de carga nos pontos excluídos.

volume_redistribution: impacto da redistribuição ótima de volumes vs. roteamento atual (redução ou incremento de custo conforme o sinal do delta); novo custo total; comparação com cenário base se disponível.

demand_change: para cada terminal com ajuste, converta o multiplicador em variação absoluta de toneladas (ex: "+X mil t/mês no Terminal TA, −Y mil t/mês no Terminal TB"); depois informe o delta líquido combinado (se os ajustes se opõem, explicite o sinal de cada um antes de combinar); delta de frota vs. cenário base; novo custo total.

demand_change_redistribution: abra com o volume adicional por terminal em toneladas absolutas e total combinado; apresente frota e custo resultantes da redistribuição ótima com delta vs. cenário base; se algum terminal operar próximo ou acima de sua capacidade, mencione o percentual de utilização e o gap.

volume_cap_redistribution: um terminal teve sua capacidade de recebimento limitada por uma restrição de volume (ex: TB limitado a 85%), e o solver redistribuiu o volume excedente otimamente entre os demais terminais. Abra com o terminal limitado e o volume total redirecionado em toneladas; depois indique para quais terminais o volume foi distribuído (em toneladas absolutas); finalize com frota e custo resultantes vs. cenário base.

capacity_overflow: este cenário rodou SEM restrições de capacidade para gerar um plano de atuação. Se há ajuste de demanda ativo, abra com o volume adicional total antes de apresentar o overflow. Destaque:
(a) Quais terminais estão acima da capacidade e em quanto (toneladas e %).
(b) Custo e frota no cenário sem restrição (o que seria necessário operacionalmente).
(c) Framing como mapa de atuação: "para absorver esse volume, Terminal TA precisaria de +X t/mês — seja por expansão de capacidade, ajuste operacional ou outra medida".
(d) Se mais de um terminal está em overflow, priorize pelo maior gap absoluto.
Priorize o maior gap; o limite de 75 palavras é estrito — cite no máximo um terminal e uma medida de atuação.
Não use "inviável" ou "sem solução" — o solver rodou; os dados são reais.

parametric_whatif: efeito primário na frota; efeito secundário ou contraintuitivo se presente; delta de custo vs. cenário base quando disponível — somente números absolutos quando não há baseline.

FORMATO: responda em 2 frases corridas; use uma 3ª frase APENAS em um destes dois casos: (a) a frota caiu mas o custo subiu, ou vice-versa, ou (b) um terminal ou ponto de coleta ultrapassou ou está em alerta de capacidade (≥95%). Máximo de 75 palavras, sem bullets, sem títulos e sem recomendação genérica. Escolha os 2 ou 3 números que explicam a decisão. Tom direto e profissional, como um colega sênior que domina o assunto. Responda sempre em português do Brasil."""


_SYSTEM_PROMPT_EN = """You are a transportation planning expert who acts as an analytical partner for logistics planners. Your job is to turn numerical optimization results into clear operational analysis, like a senior colleague explaining the scenario to the team.

COMMUNICATION PRINCIPLES:
1. Explain causes only from the "Computed facts", "Allowed explanations", and "Forbidden explanations" blocks. Do not attribute causality to capacity, distance, redistribution, bottlenecks, or a specific terminal unless explicitly allowed.
2. Operational language only. Never mention "solver", "MILP", "objective function", "variables", or "optimizer". Use: "minimum fleet", "operational cost", "collection points served".
3. Always quantify both fleet and cost. The planner needs both to make a decision.
4. Convert percentages to concrete counts. Never say just "70% of points" — say "70% of collection points, i.e. X out of Y available points".
5. Compare to baseline when data is available. Use "vs. baseline", "delta of $Y million", "X fewer trucks".
6. Explain counterintuitive results before the numbers. If fleet dropped but cost rose, explain why first.
7. Context before numbers. Sequence: what changed → primary effect → secondary or offsetting effect (if any) → net result.
8. Financial values in millions. Always "$X million/month". For deltas below $100k, use "$Xk/month". All monetary figures refer to monthly costs — append "/month" immediately after every dollar amount, without exception.
9. For cost-component scenarios (fuel, maintenance, overtime): state the component's weight in total cost before quantifying the absolute impact.
10. Never use "savings", "save", "benefit", "gain", or "upside". A cost delta is always a reduction or an increase — use "cost reduction" when the delta is negative and "cost increase" when it is positive.
11. The "availability" parameter (truck_availability) refers to the truck asset — the fraction of time the vehicle is operational and ready for use. Never interpret it as driver availability or driver shift coverage.
12. If driver wage or drivers per truck changed materially, explain that fixed-cost impact before looking for other causes.
13. When availability changes from baseline and the computed facts include a maintenance cost impact, you MUST include the impact statement in the response. Format: "Supporting X% availability adds $Yk in maintenance costs" (or "reduces" when the delta is negative). Use "$Y million" if the absolute value is ≥ $100k.

SCENARIO TYPES AND WHAT TO HIGHLIGHT (derive the type from the scenario context block in the prompt to select the right talking points):

baseline: optimal fleet for the current network + total cost.

no_baseline_whatif: no baseline has been calculated in this session — report absolute numbers only (fleet, total cost, points served); no fleet delta, no cost delta, no implied or explicit comparison against any prior reference.

two_level_cost_diff: translate both coverage levels to counts ("X of Y points"); cost delta between levels; fleet delta between levels. Never compare either level against the session baseline — compare only the target level against the origin level.

terminal_closure: which terminal closed; fleet impact (Δ vs. baseline); new total cost.

maximize_coverage_budget: compare budget to baseline cost; achievable count ("X of Y points"); excluded points if relevant.

minimize_fleet: translate % to count; minimum fleet achieved; total cost for that fleet.

minimize_cost_coverage: translate % to count; minimum cost achieved; implication for load origination at excluded points.

volume_redistribution: cost impact of optimal volume redistribution vs. current routing (cost reduction or increase depending on the sign of the delta); new total cost; comparison to baseline if available.

demand_change: for each terminal with an adjustment, convert the multiplier to absolute tonnage change (e.g. "+X thousand t/month at Terminal TA, −Y thousand t/month at Terminal TB"); then state the combined net delta (if adjustments offset each other, state the sign of each before combining); fleet delta vs. baseline; new total cost.

demand_change_redistribution: open with the additional volume per terminal in absolute tonnes and the combined total; present the fleet and cost from the optimal redistribution with a delta vs. baseline; if any terminal operates at or near capacity, state its utilisation percentage and the gap.

volume_cap_redistribution: a terminal had its incoming volume capped (e.g., TB limited to 85%) and the solver optimally redistributed the excess to other terminals. Open with the capped terminal and the total tonnes redirected; then state where the volume went (absolute tonnes per terminal); close with fleet and cost vs. baseline.

capacity_overflow: this scenario ran WITHOUT capacity constraints to produce an action plan. If demand multipliers are active, open with the total additional volume before presenting the overflow. Highlight:
(a) Which terminals are over-capacity and by how much (tonnes and %).
(b) Fleet and cost in the unconstrained scenario (what would be operationally required).
(c) Frame as an action map: "to absorb this volume, Terminal TA would need +X t/month — whether through capacity expansion, operational improvements, or other measures".
(d) If more than one terminal is in overflow, prioritise by largest absolute gap.
Prioritise the largest gap; the 75-word limit is strict — cite at most one terminal and one action measure.
Do not use "infeasible" or "no solution" — the solver ran; the data is real.

parametric_whatif: primary fleet effect; secondary or counterintuitive effect if present; cost delta vs. baseline when available — absolute numbers only when no baseline exists.

FORMAT: respond in 2 continuous sentences; use a 3rd sentence ONLY in one of these two cases: (a) fleet went down but cost went up, or vice-versa, or (b) a terminal or collection point exceeded its capacity or has a near-capacity alert (≥95%). Maximum 75 words, no bullets, no headers, and no generic recommendation. Choose the 2 or 3 numbers that explain the decision. Direct and professional tone, like a senior colleague who knows the subject well. Always respond in English."""



def create_expert_agent(provider: str, model_id: str, api_key: str, language: str = "pt") -> Agent:
    """Create the Transportation Expert Agent with a cached system prompt."""
    system_prompt = _SYSTEM_PROMPT_PT if language == "pt" else _SYSTEM_PROMPT_EN
    model, agent_sys_prompt = make_model(provider, model_id, api_key, 4096, system_prompt)
    return Agent(
        model=model,
        system_prompt=agent_sys_prompt,
        tools=[],
        callback_handler=None,
    )


def run_expert_agent(
    agent: Agent,
    milp_result: MILPResult,
    params: ScenarioParams,
    language: str = "pt",
    total_cps: int = 0,
    baseline_trucks: Optional[int] = None,
    baseline_cost: Optional[float] = None,
    terminal_demand_totals: Optional[dict[str, float]] = None,
    terminal_capacities: Optional[dict[str, float]] = None,
    baseline_params: Optional[ScenarioParams] = None,
    baseline_milp_result: Optional[MILPResult] = None,
    all_cp_ids: Optional[list[str]] = None,
    cp_capacities: Optional[dict[str, float]] = None,
    cp_demands: Optional[dict[str, float]] = None,
    network: Optional[NetworkData] = None,
) -> str:
    """Generate a plain-language insight for the MILP result."""
    if not milp_result.feasible:
        prompt = _build_infeasible_prompt(milp_result, params, total_cps)
    else:
        prompt = _build_result_prompt(
            milp_result, params, total_cps, baseline_trucks, baseline_cost,
            terminal_demand_totals, terminal_capacities,
            baseline_params, baseline_milp_result,
            all_cp_ids=all_cp_ids, cp_capacities=cp_capacities, cp_demands=cp_demands,
            network=network,
        )
        try:
            from ..app.feedback import detect_scenario_type, load_examples, format_few_shot_block
            stype = detect_scenario_type(params, milp_result)
            examples = load_examples("transportation_expert", language, stype)
            few_shot_block = format_few_shot_block(examples, language)
            if few_shot_block:
                marker = "\n\nWrite a concise operational insight:"
                idx = prompt.rfind(marker)
                if idx >= 0:
                    prompt = prompt[:idx] + few_shot_block + prompt[idx:]
                else:
                    prompt += few_shot_block
        except Exception:
            pass
    result = agent(prompt)
    return str(result).strip()


def _fmt_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}/month"


def _build_grounding_sections(
    r: MILPResult,
    p: ScenarioParams,
    baseline_params: Optional[ScenarioParams],
    baseline_milp_result: Optional[MILPResult],
    terminal_demand_totals: Optional[dict[str, float]],
    terminal_capacities: Optional[dict[str, float]],
    cp_capacities: Optional[dict[str, float]] = None,
    cp_demands: Optional[dict[str, float]] = None,
    network: Optional[NetworkData] = None,
) -> str:
    facts: list[str] = []
    allowed: list[str] = []
    forbidden: list[str] = []
    is_two_level = r.cost_a is not None and r.cost_b is not None

    # Cost component fractions — always included when components are populated
    if p.fixed_cost_components and p.fixed_cost_per_truck_month > 0:
        driver_wage = p.fixed_cost_components.get("driver_wage", 0.0)
        if driver_wage > 0:
            driver_pct = driver_wage / p.fixed_cost_per_truck_month * 100
            facts.append(
                f"Driver wage: ${driver_wage:,.0f}/truck/month = {driver_pct:.0f}% of fixed cost per truck "
                f"(total fixed: ${p.fixed_cost_per_truck_month:,.0f}/truck/month)."
            )
    if p.variable_cost_components and p.variable_cost_per_km > 0:
        fuel = p.variable_cost_components.get("fuel", 0.0)
        if fuel > 0:
            fuel_pct_var = fuel / p.variable_cost_per_km * 100
            facts.append(
                f"Fuel rate: ${fuel:.4f}/km = {fuel_pct_var:.0f}% of variable cost per km "
                f"(total variable rate: ${p.variable_cost_per_km:.4f}/km)."
            )

    # CP utilization range across served CPs
    if cp_capacities and cp_demands and r.served_cps:
        cp_utils = []
        for cp in r.served_cps:
            cap = cp_capacities.get(cp, 0.0)
            if cap <= 0:
                continue
            vol = sum(r.volumes[cp].values()) if (r.volumes and cp in r.volumes) else cp_demands.get(cp, 0.0)
            if vol > 0:
                cp_utils.append(vol / cap * 100)
        if cp_utils:
            facts.append(
                f"Collection point utilisation range ({len(cp_utils)} served CPs): "
                f"{min(cp_utils):.0f}% to {max(cp_utils):.0f}%."
            )

    if is_two_level:
        truck_delta = (r.trucks_b - r.trucks_a) if r.trucks_a is not None and r.trucks_b is not None else 0
        cost_delta = r.cost_difference if r.cost_difference is not None else r.cost_b - r.cost_a
        count_a = p.coverage_count_a or 0
        count_b = p.coverage_count_b or 0
        facts.extend([
            f"Two-level comparison only: Coverage-from is {count_a} CPs (origin level) and Coverage-to is {count_b} CPs (target level).",
            f"Fleet delta (target minus origin): {truck_delta:+d} trucks.",
            f"Cost delta (target minus origin): {_fmt_money(cost_delta)}.",
        ])
        allowed.append("two_level_coverage_comparison: compare the target coverage level only against the origin coverage level, preserving the user's requested direction.")
        forbidden.append("Do not compare this two-level coverage question against the scenario baseline; compare only target minus origin.")
        forbidden.append("Do not call the target coverage a reduction or saving versus baseline; it is only target minus origin.")

    if not is_two_level and baseline_milp_result is None:
        forbidden.append(
            "No baseline result is available for this session. "
            "Do not invent, estimate, or imply any baseline values, fleet deltas, or cost deltas versus a baseline. "
            "Report only the absolute numbers of the current scenario."
        )

    if not is_two_level and baseline_milp_result is not None and baseline_milp_result.feasible:
        truck_delta = r.trucks - baseline_milp_result.trucks
        cost_delta = r.total_cost - baseline_milp_result.total_cost
        fixed_delta = r.fixed_cost - baseline_milp_result.fixed_cost
        variable_delta = r.variable_cost - baseline_milp_result.variable_cost
        overtime_delta = r.overtime_cost_total - baseline_milp_result.overtime_cost_total
        km_delta = r.total_km - baseline_milp_result.total_km
        facts.extend([
            f"Fleet delta vs baseline: {truck_delta:+d} trucks.",
            f"Total cost delta vs baseline: {_fmt_money(cost_delta)}.",
            f"Fixed-cost delta: {_fmt_money(fixed_delta)}.",
            f"Variable-cost delta: {_fmt_money(variable_delta)}.",
            f"Overtime-cost delta: {_fmt_money(overtime_delta)}.",
            f"Distance delta: {km_delta:+,.0f} km/month.",
        ])

        drivers = [
            ("fixed_cost_delta", fixed_delta),
            ("variable_cost_delta", variable_delta),
            ("overtime_cost_delta", overtime_delta),
        ]

        if baseline_params is not None:
            base_driver = baseline_params.fixed_cost_components.get("driver_wage", 0.0)
            curr_driver = p.fixed_cost_components.get("driver_wage", 0.0)
            base_driver_total = base_driver * baseline_milp_result.trucks
            curr_driver_total = curr_driver * r.trucks
            driver_delta = curr_driver_total - base_driver_total
            non_driver_delta = fixed_delta - driver_delta
            facts.extend([
                f"Driver wage per truck: ${base_driver:,.2f}/month baseline -> ${curr_driver:,.2f}/month scenario.",
                f"Driver wage total impact: {_fmt_money(driver_delta)}.",
                f"Other fixed-cost impact: {_fmt_money(non_driver_delta)}.",
            ])
            drivers.extend([
                ("driver_wage_delta", driver_delta),
                ("non_driver_fixed_delta", non_driver_delta),
            ])
            if abs(driver_delta) >= max(100_000, abs(cost_delta) * 0.35):
                allowed.append(
                    "driver_wage_increase_is_primary_driver: explain the cost increase mainly through driver wage / drivers per truck impact."
                    if driver_delta > 0 else
                    "driver_wage_reduction_is_primary_driver: explain the cost reduction mainly through driver wage / drivers per truck impact."
                )
            if curr_driver != base_driver:
                forbidden.append("Do not ignore driver wage; it changed materially and must be discussed.")

            # Operational parameter deltas — expose direction so the expert cannot invert them
            payload_changed = p.payload != baseline_params.payload
            avail_changed = p.availability != baseline_params.availability
            if payload_changed:
                pdelta = p.payload - baseline_params.payload
                ppct = pdelta / baseline_params.payload * 100
                facts.append(
                    f"Payload per truck: {baseline_params.payload:.0f}t baseline → {p.payload:.0f}t scenario "
                    f"({pdelta:+.1f}t, {ppct:+.1f}%, {'decrease' if pdelta < 0 else 'increase'})."
                )
                if pdelta < 0:
                    forbidden.append(
                        f"Do not say payload increased; it decreased from "
                        f"{baseline_params.payload:.0f}t to {p.payload:.0f}t."
                    )
            if avail_changed:
                base_avail_pct = baseline_params.availability * 100
                curr_avail_pct = p.availability * 100
                adelta_pp = curr_avail_pct - base_avail_pct
                direction = "improvement" if adelta_pp > 0 else "reduction"
                facts.append(
                    f"Truck availability: {base_avail_pct:.0f}% baseline → {curr_avail_pct:.0f}% scenario "
                    f"({adelta_pp:+.0f} percentage points, {direction})."
                )
                if adelta_pp > 0:
                    forbidden.append(
                        f"Do not say truck availability dropped or fell; it improved from "
                        f"{base_avail_pct:.0f}% to {curr_avail_pct:.0f}%."
                    )
                else:
                    forbidden.append(
                        f"Do not say truck availability improved; it decreased from "
                        f"{base_avail_pct:.0f}% to {curr_avail_pct:.0f}%."
                    )
                if network and network.availability_sensitivity and r.total_km > 0:
                    maint_adj_per_km = 0.0
                    for cost_key, sens_per_pp in network.availability_sensitivity.items():
                        baseline_comp = network.variable_cost_components.get(cost_key, 0.0)
                        current_comp = p.variable_cost_components.get(cost_key, baseline_comp)
                        maint_adj_per_km += current_comp * adelta_pp * sens_per_pp
                    maint_cost_delta = maint_adj_per_km * r.total_km
                    maint_direction = "increase" if maint_cost_delta > 0 else "reduction"
                    maint_abs = abs(maint_cost_delta)
                    maint_fmt = (
                        f"${maint_abs / 1_000_000:.2f}M" if maint_abs >= 100_000
                        else f"${maint_abs / 1_000:.1f}k"
                    )
                    facts.append(
                        f"Maintenance cost sensitivity: {curr_avail_pct:.0f}% availability → "
                        f"maintenance cost {maint_direction} of {maint_fmt}/month "
                        f"(${maint_cost_delta:+,.0f}/month vs baseline)."
                    )
                    allowed.append(
                        f"maintenance_sensitivity_disclosure [MANDATORY]: state that supporting "
                        f"{curr_avail_pct:.0f}% availability "
                        f"{'adds' if maint_cost_delta > 0 else 'reduces'} {maint_fmt}/month "
                        "in maintenance costs. This statement must appear in the response."
                    )
            if payload_changed and avail_changed and p.payload < baseline_params.payload and p.availability > baseline_params.availability:
                allowed.append(
                    "payload_loss_with_availability_compensation: payload per truck decreased (fewer tonnes per trip) "
                    "while availability improved (more uptime per truck as a partial offset); "
                    "the user is modelling a deteriorating-payload scenario where higher uptime is used to compensate, "
                    "but the net fleet and cost outcome shows whether that offset was sufficient — "
                    "and the higher availability typically requires more maintenance, raising costs."
                )

        primary_name, primary_value = max(drivers, key=lambda item: abs(item[1]))
        facts.append(f"Largest computed cost bridge item: {primary_name} ({_fmt_money(primary_value)}).")
        if primary_name == "variable_cost_delta":
            allowed.append("variable_cost_delta_is_primary_driver: discuss distance/km or variable-cost impact as the main cost driver.")
        elif primary_name == "fixed_cost_delta":
            allowed.append("fixed_cost_delta_is_primary_driver: discuss fixed fleet cost as the main cost driver.")
        elif primary_name == "overtime_cost_delta":
            allowed.append("overtime_cost_delta_is_primary_driver: discuss overtime as the main cost driver.")

        if abs(variable_delta) < abs(cost_delta) * 0.25:
            forbidden.append("Do not attribute the total cost movement mainly to distance or variable cost.")
        if truck_delta < 0 and cost_delta > 0:
            allowed.append("fleet_down_cost_up_tradeoff: explain that fewer trucks did not reduce total cost because another cost component rose more.")

    if p.terminal_volume_caps and p.volume_redistribution:
        capped = ", ".join(
            f"Terminal {t} (capped to {v * 100:.0f}%)"
            for t, v in p.terminal_volume_caps.items()
        )
        allowed.append(
            f"terminal_volume_cap_redistribution: {capped} had its incoming volume capped; "
            "discuss the absolute tonnes redirected to other terminals and the resulting fleet/cost outcome. "
            "Use the 'Volume cap redistribution' section for exact figures."
        )
        allowed.append("volume_redistribution_was_requested: it is allowed to discuss route or volume redistribution.")
    elif p.volume_redistribution:
        allowed.append("volume_redistribution_was_requested: it is allowed to discuss route or volume redistribution.")
    else:
        forbidden.append("Do not claim that volume was redistributed or rerouted; volume_redistribution is false.")

    if terminal_demand_totals and terminal_capacities:
        terminal_lines = []
        terminal_over_capacity = []
        terminal_near_capacity: list[tuple[str, float, float, float]] = []
        # When the solver ran without capacity constraints (action-map run), r.terminal_overflows
        # contains the ground-truth volumes actually routed to each terminal after redistribution.
        # Using terminal_demand_totals × multiplier would be wrong here: redistribution shifts
        # volume away from historical splits, causing systematic over- or under-estimation.
        use_solver_volumes = p.skip_capacity_constraints and bool(r.terminal_overflows)
        # For volume_cap_redistribution, terminal_demand_multipliers is empty (the cap is in
        # terminal_volume_caps instead), so the fallback formula would return raw historical demand.
        # Read actual solver-distributed volumes from r.volumes to get the correct post-cap figures.
        use_redistribution_volumes = (
            p.volume_redistribution and bool(r.volumes)
        )
        terminal_received_from_solver: dict[str, float] = {}
        if use_redistribution_volumes:
            for cp_vols in r.volumes.values():
                for t_id, vol in cp_vols.items():
                    terminal_received_from_solver[t_id] = (
                        terminal_received_from_solver.get(t_id, 0.0) + vol
                    )
        for tid in terminal_demand_totals:
            cap = terminal_capacities.get(tid, 0.0)
            if cap <= 0:
                continue
            if use_solver_volumes:
                if tid not in r.terminal_overflows:
                    # Not in overflows → definitively within capacity in the solver result.
                    forbidden.append(f"Do not say Terminal {tid} is over capacity or critical; it is within capacity in the solver result.")
                    continue
                eff = r.terminal_overflows[tid]
            elif use_redistribution_volumes:
                eff = terminal_received_from_solver.get(tid, 0.0)
            else:
                eff = terminal_demand_totals[tid] * p.terminal_demand_multipliers.get(tid, 1.0)
            util = eff / cap
            terminal_lines.append(f"Terminal {tid}: {eff:,.0f} t/month effective, capacity {cap:,.0f}, utilisation {util*100:.0f}%.")
            if eff > cap:
                terminal_over_capacity.append((tid, eff, cap, util))
            elif util >= 0.95:
                terminal_near_capacity.append((tid, eff, cap, util))
            else:
                forbidden.append(f"Do not say Terminal {tid} is over capacity or critical; utilisation is {util*100:.0f}%.")
        if terminal_lines:
            facts.append("Terminal capacity status: " + " ".join(terminal_lines))
        if terminal_over_capacity:
            allowed.append("terminal_capacity_issue_detected: discuss only the terminals listed as over capacity in Computed facts.")
        else:
            forbidden.append("Do not diagnose a terminal capacity bottleneck; no terminal is above capacity.")
        if terminal_near_capacity:
            near_lines = [
                f"Terminal {tid}: {eff:,.0f} t/month effective, capacity {cap:,.0f}, utilisation {util*100:.0f}% — ALERT: near capacity (≥95%)."
                for tid, eff, cap, util in terminal_near_capacity
            ]
            facts.append("Near-capacity alert (≥95%): " + " ".join(near_lines))
            allowed.append("terminal_near_capacity_alert: mention that the listed terminal(s) are approaching the capacity limit (≥95% utilisation); do not treat as overflow.")
        else:
            forbidden.append("Do not report a near-capacity alert; no terminal reached 95% utilisation.")

    if r.cp_overflows:
        cp_lines = []
        for cp, eff in sorted(r.cp_overflows.items()):
            cp_lines.append(f"{cp}: {eff:,.0f} t/month routed/effective.")
        facts.append("CP capacity action map is present for: " + "; ".join(cp_lines))
        allowed.append("cp_capacity_overflow_detected: discuss collection point capacity only if needed; do not convert it into a terminal issue.")
    elif p.skip_capacity_constraints:
        forbidden.append("Do not claim collection point overflow unless CP overflow is listed.")

    if not allowed:
        allowed.append("summarize_numeric_deltas_only: explain only the quantified fleet, cost, coverage, and parameter changes.")
    if not forbidden:
        forbidden.append("No extra causal speculation beyond computed facts.")
    forbidden.append(
        "Do not introduce analytical concepts, thresholds, or benchmarks (such as structural slack, "
        "sustainable utilisation floors, capacity buffers, or any similar derived metric) "
        "that are not explicitly listed in Computed facts."
    )

    return (
        "\n\nComputed facts:\n  - " + "\n  - ".join(facts or ["No additional computed facts."])
        + "\n\nAllowed explanations:\n  - " + "\n  - ".join(allowed)
        + "\n\nForbidden explanations:\n  - " + "\n  - ".join(forbidden)
    )


def _build_result_prompt(
    r: MILPResult,
    p: ScenarioParams,
    total_cps: int,
    baseline_trucks: Optional[int],
    baseline_cost: Optional[float],
    terminal_demand_totals: Optional[dict[str, float]] = None,
    terminal_capacities: Optional[dict[str, float]] = None,
    baseline_params: Optional[ScenarioParams] = None,
    baseline_milp_result: Optional[MILPResult] = None,
    all_cp_ids: Optional[list[str]] = None,
    cp_capacities: Optional[dict[str, float]] = None,
    cp_demands: Optional[dict[str, float]] = None,
    network: Optional[NetworkData] = None,
) -> str:
    served = len(r.served_cps)
    pct_served = f"{served / total_cps * 100:.0f}%" if total_cps > 0 else "N/A"
    closed_terminals = [tid for tid, active in p.terminals_active.items() if not active]

    # Raw scenario context — Expert derives the type from these fields
    scenario_context = (
        f"<scenario_context>"
        f"\n  <is_baseline>{str(p.is_baseline).lower()}</is_baseline>"
        f"\n  <terminals_closed>{', '.join(closed_terminals) if closed_terminals else 'none'}</terminals_closed>"
        f"\n  <terminal_demand_multipliers>{dict(p.terminal_demand_multipliers) or 'none'}</terminal_demand_multipliers>"
        f"\n  <terminal_volume_caps>{dict(p.terminal_volume_caps) or 'none'}</terminal_volume_caps>"
        f"\n  <volume_redistribution>{str(p.volume_redistribution).lower()}</volume_redistribution>"
        f"\n  <skip_capacity_constraints>{str(p.skip_capacity_constraints).lower()}</skip_capacity_constraints>"
        f"\n  <objective>{p.objective}</objective>"
        f"\n  <min_coverage_count>{p.min_coverage_count if p.min_coverage_count is not None else 'none'}</min_coverage_count>"
        f"\n  <budget>{p.budget if p.budget is not None else 'none'}</budget>"
        f"\n  <coverage_count_a>{p.coverage_count_a if p.coverage_count_a is not None else 'none'}</coverage_count_a>"
        f"\n  <coverage_count_b>{p.coverage_count_b if p.coverage_count_b is not None else 'none'}</coverage_count_b>"
        f"\n</scenario_context>"
    )

    # Delta vs. baseline
    delta_section = ""
    is_two_level = r.cost_a is not None and r.cost_b is not None
    if not is_two_level and baseline_trucks is not None and baseline_cost is not None:
        delta_trucks = r.trucks - baseline_trucks
        delta_cost = r.total_cost - baseline_cost
        sign_t = "+" if delta_trucks >= 0 else ""
        sign_c = "+" if delta_cost >= 0 else ""
        delta_section = (
            f"\nBaseline comparison:"
            f"\n  Baseline trucks: {baseline_trucks}"
            f"\n  Delta trucks: {sign_t}{delta_trucks}"
            f"\n  Baseline cost: ${baseline_cost:,.0f}/month"
            f"\n  Delta cost: {sign_c}${delta_cost:,.0f}/month"
        )

    # Two-level cost difference (scenario type two_level_cost_diff)
    two_level_section = ""
    if is_two_level:
        diff = r.cost_difference or 0.0
        truck_diff = (r.trucks_b - r.trucks_a) if r.trucks_a is not None and r.trucks_b is not None else 0
        count_a = p.coverage_count_a or 0
        count_b = p.coverage_count_b or 0
        pct_a = f"{count_a / total_cps * 100:.0f}%" if total_cps > 0 else ""
        pct_b = f"{count_b / total_cps * 100:.0f}%" if total_cps > 0 else ""
        two_level_section = (
            f"\nTwo-level coverage comparison:"
            f"\n  Coverage-from (origin): {count_a} of {total_cps} CPs ({pct_a}) — cost ${r.cost_a:,.0f}/month"
            f"\n  Coverage-to (target): {count_b} of {total_cps} CPs ({pct_b}) — cost ${r.cost_b:,.0f}/month"
            f"\n  Fleet difference (target minus origin): {truck_diff:+d} trucks"
            f"\n  Cost difference (target minus origin): ${diff:,.0f}/month"
        )

    # Cost component breakdown with percentage share
    cost_section = (
        f"\nCost breakdown:"
        f"\n  Fixed (trucks): ${r.fixed_cost:,.0f}/month"
        f"\n  Variable (km): ${r.variable_cost:,.0f}/month"
        f"\n  Overtime: ${r.overtime_cost_total:,.0f}/month"
    )
    if r.total_cost > 0:
        cost_section += (
            f"\n  Fixed share: {r.fixed_cost / r.total_cost * 100:.1f}%"
            f"\n  Variable share: {r.variable_cost / r.total_cost * 100:.1f}%"
            f"\n  Overtime share: {r.overtime_cost_total / r.total_cost * 100:.1f}%"
        )

    # Fuel fraction of total cost (for cost-component scenarios)
    fuel_section = ""
    fuel_per_km = p.variable_cost_components.get("fuel", 0.0) if p.variable_cost_components else 0.0
    if fuel_per_km > 0 and p.variable_cost_per_km > 0 and r.total_cost > 0:
        fuel_fraction_of_variable = fuel_per_km / p.variable_cost_per_km
        fuel_dollars = fuel_fraction_of_variable * r.variable_cost
        fuel_pct_of_total = fuel_dollars / r.total_cost * 100
        fuel_section = (
            f"\nFuel cost context:"
            f"\n  Fuel rate: ${fuel_per_km:.2f}/km out of ${p.variable_cost_per_km:.2f}/km total variable rate"
            f"\n  Estimated fuel share of total operational cost: {fuel_pct_of_total:.1f}%"
        )

    # Demand change context — always show absolute tonnage, never just the multiplier
    demand_section = ""
    if p.terminal_demand_multipliers:
        parts = []
        for tid, mul in p.terminal_demand_multipliers.items():
            pct_change = (mul - 1.0) * 100
            sign = "+" if pct_change >= 0 else ""
            baseline_vol = (terminal_demand_totals or {}).get(tid, 0.0)
            if baseline_vol > 0:
                abs_change = baseline_vol * (mul - 1.0)
                effective_vol = baseline_vol * mul
                parts.append(
                    f"  Terminal {tid}: {sign}{pct_change:.0f}% demand"
                    f" ({baseline_vol:,.0f} t/month baseline"
                    f" -> {effective_vol:,.0f} t/month effective,"
                    f" change of {abs_change:+,.0f} t/month)"
                )
            else:
                parts.append(
                    f"  Terminal {tid}: {sign}{pct_change:.0f}% demand change (multiplier {mul:.2f})"
                )
        demand_section = "\nDemand changes applied:\n" + "\n".join(parts)

    volume_cap_section = ""
    if p.terminal_volume_caps and p.volume_redistribution:
        demand_totals = terminal_demand_totals or {}
        cap_parts = []
        for tid, cap_frac in p.terminal_volume_caps.items():
            hist_vol = demand_totals.get(tid, 0.0)
            pct_cap = cap_frac * 100
            if hist_vol > 0:
                max_allowed = hist_vol * cap_frac
                redirected_vol = hist_vol * (1.0 - cap_frac)
                cap_parts.append(
                    f"  Terminal {tid}: capped to {pct_cap:.0f}% of historical"
                    f" ({hist_vol:,.0f} t/month baseline → max {max_allowed:,.0f} t/month;"
                    f" {redirected_vol:,.0f} t/month redirected)"
                )
            else:
                cap_parts.append(f"  Terminal {tid}: capped to {pct_cap:.0f}% of historical volume")
        if r.volumes:
            terminal_received: dict[str, float] = {}
            for cp_vols in r.volumes.values():
                for t_id, vol in cp_vols.items():
                    terminal_received[t_id] = terminal_received.get(t_id, 0.0) + vol
            recv_parts = []
            for t_id in sorted(demand_totals):
                hist = demand_totals.get(t_id, 0.0)
                received = terminal_received.get(t_id, 0.0)
                delta = received - hist
                sign = "+" if delta >= 0 else ""
                recv_parts.append(
                    f"  Terminal {t_id}: {received:,.0f} t/month received"
                    f" ({sign}{delta:,.0f} vs historical {hist:,.0f} t/month)"
                )
            if recv_parts:
                cap_parts.append("Terminal volumes after redistribution:")
                cap_parts.extend(recv_parts)
        volume_cap_section = "\nVolume cap redistribution:\n" + "\n".join(cap_parts)

    # Capacity overflow context — only for over-capacity action-map runs
    capacity_section = ""
    if p.skip_capacity_constraints and r.terminal_overflows and terminal_capacities:
        overflow_lines = []
        for tid, eff_demand in sorted(
            r.terminal_overflows.items(),
            key=lambda kv: kv[1] - terminal_capacities.get(kv[0], 0),
            reverse=True,
        ):
            cap = terminal_capacities.get(tid, 0.0)
            if cap > 0 and eff_demand > cap:
                gap = eff_demand - cap
                pct = (eff_demand / cap - 1) * 100
                overflow_lines.append(
                    f"  Terminal {tid}: {eff_demand:,.0f} t/month routed"
                    f"  cap {cap:,.0f} t/month  gap +{gap:,.0f} t (+{pct:.1f}%)"
                )
        if overflow_lines:
            capacity_section = "\nCapacity overflow (action targets):\n" + "\n".join(overflow_lines)

    # Excluded CPs (not served) — named so the Expert can cite them explicitly
    excluded_cps_section = ""
    if all_cp_ids and not is_two_level:
        excluded = [cp for cp in all_cp_ids if cp not in set(r.served_cps)]
        if excluded:
            excluded_cps_section = f"\nExcluded collection points (not served in this scenario): {', '.join(excluded)}"

    # Terminal closure context
    terminal_section = ""
    if closed_terminals:
        terminal_section = f"\nClosed terminals: {', '.join(closed_terminals)}"

    # Budget context
    budget_section = ""
    if p.budget:
        budget_section = f"\nBudget constraint: ${p.budget:,.0f}/month"

    grounding_sections = _build_grounding_sections(
        r,
        p,
        baseline_params,
        baseline_milp_result,
        terminal_demand_totals,
        terminal_capacities,
        cp_capacities=cp_capacities,
        cp_demands=cp_demands,
        network=network,
    )

    return (
        f"{scenario_context}"
        f"\n\nResult:"
        f"\n  Trucks: {r.trucks}"
        f"\n  Collection points served: {served} of {total_cps} ({pct_served})"
        f"\n  Total monthly cost: ${r.total_cost:,.0f}"
        f"\n  Total km/month: {r.total_km:,.0f}"
        f"{cost_section}"
        f"{fuel_section}"
        f"{delta_section}"
        f"{two_level_section}"
        f"{demand_section}"
        f"{volume_cap_section}"
        f"{capacity_section}"
        f"{excluded_cps_section}"
        f"{terminal_section}"
        f"{budget_section}"
        f"{grounding_sections}"
        f"\n\nOperational parameters: payload={p.payload}t, speed_loaded={p.speed_loaded}km/h, "
        f"truck_availability={p.availability * 100:.0f}% (asset uptime — not driver availability), "
        f"overtime={p.overtime_hours}h/day, working_days={p.working_days}"
        f"\n\nWrite a concise operational insight: 2 sentences by default, 3 only if a relevant trade-off must be stated, maximum 75 words. "
        f"Use only the computed facts and allowed explanations for causal claims. "
        f"Respect every forbidden explanation. "
        f"Derive the scenario type from the context above and apply the corresponding talking points."
    )


def _build_infeasible_prompt(
    r: MILPResult, p: ScenarioParams, total_cps: int
) -> str:
    min_cov = p.min_coverage_count or total_cps or "all"
    terminal_status = ", ".join(
        f"{tid}={'yes' if active else 'no'}"
        for tid, active in p.terminals_active.items()
    )
    return (
        f"The plan has no feasible solution.\n"
        f"Reason: {r.infeasibility_reason}\n\n"
        f"Scenario parameters:\n"
        f"- Budget: {'$' + f'{p.budget:,.0f}' if p.budget else 'none'}\n"
        f"- Min coverage: {min_cov} CPs\n"
        f"- Terminals active: {terminal_status}\n\n"
        f"Write a concise 2-sentence explanation of why there is no viable plan "
        f"and suggest one concrete adjustment the planner could try. Use a 3rd sentence only if needed, maximum 75 words. "
        f"Use operational language only — no solver or technical terminology."
    )
