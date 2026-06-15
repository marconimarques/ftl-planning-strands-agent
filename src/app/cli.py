"""Main CLI — Rich terminal interface for the truck fleet planning tool."""

from __future__ import annotations

import os
import sys
import time as _time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style

# Win32-only: prompt_toolkit loses the console handle after heavy Rich output.
# Import defensively so the same code runs on non-Windows platforms.
try:
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError as _Win32ConsoleError
except ImportError:
    _Win32ConsoleError = None  # type: ignore[assignment,misc]

from rich import box as rich_box
from rich.align import Align
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..domain.data_types import MILPResult, PipelineResult, ScenarioParams
from ..domain.loader import load_network_data
from ..domain.capacity_check import check_capacity
from ..agents.or_agent import create_or_agent, run_or_agent
from ..agents.transportation_expert import create_expert_agent
from ..agents.data_expert import (
    classify_session,
    serialize_scenarios,
    run_data_expert_agent,
)
from ..agents.intent_classifier import classify_intent, create_classifier_agent
from ..agents.model_factory import MODEL_REGISTRY, get_api_key, get_agent_model
from ..agents.shock_response_agent import create_shock_response_agent, run_shock_response_agent
from .pipeline import run_pipeline, run_pipeline_from_params, run_relocation_pipeline, build_baseline_params
from ..models.solver import run_milp_solver
from .display import (
    console,
    print_commands_list,
    print_network_relocate,
    print_summary_table,
    print_param_recap,
    print_insight,
    print_infeasible_explanation,
    print_served_cps,
    print_volume_matrix,
    print_detail,
    print_message,
    print_capacity_warning,
    print_over_capacity_highlights,
    print_cp_over_capacity_highlights,
    render_data_expert,
    render_shock_response,
    MSG_OK, MSG_ERR, MSG_WARN, MSG_INFO, MSG_QUIET,
)
from .commands import (
    handle_network,
    handle_requirements,
    handle_questions,
    handle_onboarding,
    handle_limits,
    handle_lane_costs,
)
from .export import export_result
from .feedback import detect_scenario_type, save_feedback
from .i18n import t


_TRANSIENT_API_ERRORS = {
    "overloaded_error": {
        "pt": "API momentaneamente sobrecarregada. Aguarde alguns segundos e tente novamente.",
        "en": "API temporarily overloaded. Wait a few seconds and try again.",
    },
    "rate_limit_error": {
        "pt": "Limite de requisições da API atingido. Aguarde alguns segundos e tente novamente.",
        "en": "API rate limit reached. Wait a few seconds and try again.",
    },
}


def _transient_api_message(exc: Exception, language: str) -> str | None:
    """Return a user-friendly message if exc (or its cause chain) is a known transient API error."""
    cause: Exception | None = exc
    seen: set[int] = set()
    while cause is not None:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        s = str(cause)
        for key, msgs in _TRANSIENT_API_ERRORS.items():
            if key in s:
                return msgs.get(language) or msgs["en"]
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
    return None


def _print_pipeline_error(exc: Exception, language: str) -> None:
    """Print a pipeline error — friendly message for known transient API errors, full traceback otherwise."""
    friendly = _transient_api_message(exc, language)
    if friendly:
        console.print(f"[{MSG_WARN}]{friendly}[/{MSG_WARN}]")
    else:
        console.print(f"[bold red]Pipeline error: {exc}[/bold red]")
        traceback.print_exc()


_BAR_LEN = 41


class _TimedSpinner:
    """Rich renderable: two-line progress bar spinner with elapsed time."""

    def __init__(self, text: str, style: str = "dark_orange") -> None:
        self._text = text
        self._style = style
        self._start = _time.monotonic()

    def __rich_console__(self, console, options):
        import math
        elapsed = _time.monotonic() - self._start
        p = min(99, round((1 - math.exp(-elapsed / 8)) * 100))
        filled = round(p * _BAR_LEN / 100)
        yield Text.assemble(
            f"  {self._text}  ",
            (f"({int(elapsed)}s)", "dim"),
        )
        yield Text.assemble(
            "    ",
            ("▰" * filled, self._style),
            ("▱" * (_BAR_LEN - filled), "color(240)"),
            f" {p}%",
        )


_FLEET_ART = """\
 ███████╗██╗     ███████╗███████╗████████╗
 ██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝
 █████╗  ██║     █████╗  █████╗     ██║
 ██╔══╝  ██║     ██╔══╝  ██╔══╝     ██║
 ██║     ███████╗███████╗███████╗   ██║
 ╚═╝     ╚══════╝╚══════╝╚══════╝   ╚═╝   """

# Convenience dict: alias -> model_id (for display only)
MODEL_IDS = {alias: model_id for alias, (_, model_id) in MODEL_REGISTRY.items()}

DEFAULT_MODEL = "anthropic"
DEFAULT_LANGUAGE = "pt"
DEFAULT_PROMPT_MODE = "simple" if sys.platform == "win32" else "prompt_toolkit"

# (en_canonical, pt_alias, i18n_key)  — display args are for the help table only
_SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/baseline",             "/cenario-base",        "cmd_baseline"),
    ("/relocate",             "/realocar",           "cmd_relocate"),
    ("/network",              "/rede",               "cmd_network"),
    ("/network-relocate",     "/rede-realocação",    "cmd_network_relocate"),
    ("/requirements",         "/requisitos",         "cmd_requirements"),
    ("/questions",            "/perguntas",          "cmd_questions"),
    ("/onboarding",           "/introdução",         "cmd_onboarding"),
    ("/detail <model>",       "/detalhe <modelo>",   "cmd_detail"),
    ("/export",               "/exportar",           "cmd_export"),
    ("/model <provider>",     "/modelo <provedor>",  "cmd_model"),
    ("/language <code>",      "/idioma <código>",    "cmd_language"),
    ("/data-expert",          "/especialista-dados", "cmd_data_expert"),
    ("/llm on|off",           "/llm on|off",         "cmd_llm"),
    ("/limits",               "/limites",            "cmd_limits"),
    ("/lane-costs",           "/rentabilidade",      "cmd_lane_costs"),
    ("/learning on|off",     "/aprendizado on|off", "cmd_learning"),
    ("/clear",                "/limpar",             "cmd_clear"),
    ("/help",                 "/ajuda",              "cmd_help"),
    ("/quit",                 "/sair",               "cmd_quit"),
]

# Base command names (no args) for both EN and PT — used by the guardrail
_COMMAND_NAMES = frozenset(
    cmd.split()[0].lstrip("/")
    for en, pt, _ in _SLASH_COMMANDS
    for cmd in (en, pt)
)

# PT alias → EN canonical  (base names only, e.g. "/realocar" → "/relocate")
_PT_TO_EN: dict[str, str] = {
    pt.split()[0]: en.split()[0]
    for en, pt, _ in _SLASH_COMMANDS
    if pt.split()[0] != en.split()[0]
}
_PT_TO_EN.update({
    "/introducao": "/onboarding",
    "/rede-realocacao": "/network-relocate",
    "/especialista-dados": "/data-expert",
    "/aprendizado": "/learning",
    "/limites": "/limits",
})

_MENU_STYLE = Style.from_dict({
    "completion-menu.completion":              "bg:#0a2a4a #e0e0e0",
    "completion-menu.completion.current":      "bg:#1565c0 #ffffff bold",
    "completion-menu.meta.completion":         "bg:#071d33 #888888",
    "completion-menu.meta.completion.current": "bg:#1565c0 #cccccc",
    "scrollbar.background":                    "bg:#0a2a4a",
    "scrollbar.button":                        "bg:#1565c0",
})


class _SlashCompleter(Completer):
    """Shows all slash commands when '/' is typed; filters as user types more."""

    def __init__(self, get_language) -> None:
        self._get_language = get_language

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        lower = text.lower()
        language = self._get_language()
        for en_full, pt_full, i18n_key in _SLASH_COMMANDS:
            base = pt_full.split()[0] if language == "pt" else en_full.split()[0]
            if base.startswith(lower):
                yield Completion(
                    base,
                    start_position=-len(text),
                    display=base,
                    display_meta=t(i18n_key, language),
                )


class FleetPlanningCLI:
    """Main CLI session controller."""

    def __init__(self) -> None:
        self.language: str = DEFAULT_LANGUAGE
        self.model_alias: str = DEFAULT_MODEL
        self.llm_insights: bool = True
        self.feedback_enabled: bool = False
        self.query_count: int = 0
        self.baseline_result: Optional[PipelineResult] = None
        self.last_result: Optional[PipelineResult] = None
        self.scenario_history: list[PipelineResult] = []
        self.prompt_mode: str = os.environ.get("FLEET_PROMPT_MODE", DEFAULT_PROMPT_MODE).lower().strip()
        self.debug_cli: bool = os.environ.get("FLEET_DEBUG_CLI", "").lower().strip() in {"1", "true", "yes", "on"}

        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            console.print(
                "[bold red]Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY.[/bold red]"
            )
            sys.exit(1)
        # If Anthropic key is absent, fall back to a default OpenAI model
        elif not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"):
            self.model_alias = "openai"

        self.network = load_network_data()
        self._init_agents()
        self._prompt_session: Optional[PromptSession] = None
        if self.prompt_mode != "simple":
            try:
                self._prompt_session = self._make_prompt_session()
            except Exception as exc:
                self._debug_log(f"prompt_session_init_error {type(exc).__name__}: {exc}")
                self.prompt_mode = "simple"

    def _debug_log(self, message: str) -> None:
        """Append CLI lifecycle diagnostics when FLEET_DEBUG_CLI is enabled."""
        if not self.debug_cli:
            return
        try:
            log_path = Path("bugs") / "cli_debug.log"
            log_path.parent.mkdir(exist_ok=True)
            stamp = datetime.now().isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{stamp} q={self.query_count} {message}\n")
        except Exception:
            pass

    def _init_agents(self) -> None:
        provider, _ = MODEL_REGISTRY[self.model_alias]
        reasoning_model = get_agent_model(provider, "reasoning")
        api_key = get_api_key(provider)
        self.or_agent = create_or_agent(provider, reasoning_model, api_key, self.language)
        self.expert_agent = create_expert_agent(provider, reasoning_model, api_key, self.language)
        self.classifier_agent = create_classifier_agent(provider, api_key, self.language)

    def _make_prompt_session(self) -> PromptSession:
        return PromptSession(
            completer=_SlashCompleter(lambda: self.language),
            complete_while_typing=True,
            style=_MENU_STYLE,
            reserve_space_for_menu=8,
        )

    def _flush_console_input(self) -> None:
        """Clear stale Win32 console events left by Rich/prompt_toolkit output."""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            std_input_handle = kernel32.GetStdHandle(-10)
            if std_input_handle not in (0, -1):
                kernel32.FlushConsoleInputBuffer(std_input_handle)
        except Exception:
            pass

    def _prepare_next_prompt(self) -> None:
        """Reset terminal state before the next prompt is displayed."""
        self._debug_log("prepare_next_prompt:start")
        try:
            console.file.flush()
        except Exception:
            pass

        self._flush_console_input()
        self._debug_log("prepare_next_prompt:end")

    def _read_prompt(self) -> str:
        """Read one user command or natural-language query."""
        prompt_str = ("Usuário> " if self.language == "pt" else "User> ")
        self._debug_log(f"read_prompt:start mode={self.prompt_mode}")
        if self.prompt_mode == "simple":
            value = input(prompt_str)
            self._debug_log("read_prompt:end simple")
            return value
        try:
            if self._prompt_session is None:
                self._prompt_session = self._make_prompt_session()
            value = self._prompt_session.prompt(prompt_str)
            self._debug_log("read_prompt:end prompt_toolkit")
            return value
        except Exception as exc:
            self._debug_log(f"read_prompt:prompt_toolkit_error {type(exc).__name__}: {exc}")
            self.prompt_mode = "simple"
            self._prompt_session = None
            print_message(
                "Prompt avançado indisponível; alternando para modo simples.",
                style=MSG_WARN,
            )
            value = input(prompt_str)
            self._debug_log("read_prompt:end simple_after_prompt_toolkit_error")
            return value

    def _print_banner(self) -> None:
        """Render the startup banner."""
        if self.language == "pt":
            tagline = "Suporte ao Dimensionamento de Frota Própria - Carga Lotação"
            tips = (
                "Dicas para começar:\n"
                '1. Use o comando "/introdução" e rapidamente entenda a ferramenta.\n'
                '2. Rode o cenário base e construa novos cenários "E se...".\n'
                '3. Sempre que precisar recorra ao comando "/ajuda".'
            )
        else:
            tagline = "Fleet Sizing Support - Full Truck Load (FTL)"
            tips = (
                "Getting started:\n"
                '1. Run "/onboarding" to quickly understand the tool.\n'
                '2. Run the baseline scenario and build new "What if?" scenarios.\n'
                '3. Whenever you need help, use the "/help" command.'
            )

        art = Text(_FLEET_ART, style="dark_orange", no_wrap=True)

        sub = Text()
        sub.append("P  L  A  N  N  I  N  G", style="color(244) bold")

        tag = Text()
        tag.append(f"\n{tagline}", style="color(250)")

        body = Text.assemble(art, "\n", sub, tag, "\n\n")
        body.append(tips, style="color(244)")

        console.print(
            Panel(
                Align.center(body),
                border_style="dark_orange",
                box=rich_box.ROUNDED,
                padding=(0, 2),
            )
        )
        console.print()

    def run(self) -> None:
        """Main session loop."""
        self._print_banner()
        while True:
            try:
                self._debug_log("loop:before_read_prompt")
                user_input = self._read_prompt().strip()
                self._debug_log("loop:after_read_prompt")
            except (EOFError, KeyboardInterrupt):
                console.print()
                print_message(t("quit_msg", self.language), style=MSG_QUIET)
                break
            except Exception:
                self._debug_log("loop:read_prompt_unhandled_exception")
                # prompt_toolkit's Win32 output caches the console handle; after
                # heavy Rich output the handle can become stale. Re-creating the
                # session calls GetStdHandle() fresh and recovers cleanly.
                # _make_prompt_session itself can fail if the handle is still
                # transitioning — wrap it so the loop always continues.
                if self.prompt_mode != "simple":
                    try:
                        self._prompt_session = self._make_prompt_session()
                    except Exception:
                        pass
                continue

            if not user_input:
                continue

            try:
                if user_input.startswith("/"):
                    should_continue = self._handle_command(user_input)
                    if not should_continue:
                        break
                else:
                    first_word = user_input.split(maxsplit=1)[0].lower()
                    if first_word in _COMMAND_NAMES:
                        print_message(
                            t("possible_command", self.language, cmd=first_word),
                            style=MSG_WARN,
                        )
                    else:
                        self._debug_log("loop:handle_query:start")
                        self._handle_query(user_input)
                        self._debug_log("loop:handle_query:end")
            except Exception as exc:
                console.print(f"[bold red]Erro interno: {exc}[/bold red]")
                import traceback
                traceback.print_exc()
            finally:
                self._debug_log("loop:finally_prepare:start")
                self._prepare_next_prompt()
                self._debug_log("loop:finally_prepare:end")

    def _handle_command(self, cmd_input: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        parts = cmd_input.split(maxsplit=1)
        cmd = _PT_TO_EN.get(parts[0].lower(), parts[0].lower())
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/", "/help"):
            rows = [
                (pt if self.language == "pt" else en, t(key, self.language))
                for en, pt, key in _SLASH_COMMANDS
            ]
            print_commands_list(self.language, rows)

        elif cmd == "/quit":
            print_message(t("quit_msg", self.language), style=MSG_QUIET)
            return False

        elif cmd == "/baseline":
            self._handle_baseline()

        elif cmd == "/relocate":
            self._handle_relocate()

        elif cmd == "/network":
            handle_network(self.network, self.language)

        elif cmd == "/network-relocate":
            self._handle_network_relocate()

        elif cmd == "/requirements":
            handle_requirements(self.network, self.language)

        elif cmd == "/limits":
            handle_limits(self.network, self.language)

        elif cmd == "/lane-costs":
            self._handle_lane_costs()

        elif cmd == "/questions":
            handle_questions(self.network, self.language)

        elif cmd == "/onboarding":
            handle_onboarding(self.network, self.language)

        elif cmd == "/detail":
            self._handle_detail(args)

        elif cmd == "/export":
            self._handle_export()

        elif cmd == "/model":
            self._handle_model(args)

        elif cmd == "/language":
            self._handle_language(args)

        elif cmd == "/data-expert":
            self._cmd_data_expert()

        elif cmd == "/llm":
            self._handle_llm(args)

        elif cmd == "/learning":
            self._handle_learning(args)

        elif cmd == "/clear":
            os.system("cls" if sys.platform == "win32" else "clear")

        else:
            print_message(t("unknown_command", self.language, cmd=cmd), style=MSG_ERR)

        return True

    def _ask_baseline_confirmation(self, language: str) -> bool:
        """Interactive yes/no prompt when a new scenario is requested without a baseline."""
        prompt_str = t("baseline_confirm_prompt", language)
        yes_key = t("baseline_confirm_yes", language)
        try:
            if self.prompt_mode == "simple":
                answer = input(prompt_str).strip().lower()
            else:
                if self._prompt_session is None:
                    self._prompt_session = self._make_prompt_session()
                answer = self._prompt_session.prompt(prompt_str).strip().lower()
            return answer == yes_key
        except Exception:
            return False

    def _ask_show_tables(self) -> bool:
        """Prompt user whether to display table breakdown after an LLM insight."""
        prompt_str = t("show_tables_prompt", self.language)
        yes_key = t("show_tables_yes", self.language)
        try:
            if self.prompt_mode == "simple":
                answer = input(prompt_str).strip().lower()
            else:
                if self._prompt_session is None:
                    self._prompt_session = self._make_prompt_session()
                answer = self._prompt_session.prompt(prompt_str).strip().lower()
            return answer == yes_key
        except Exception:
            return True

    def _ask_over_capacity_confirmation(self, language: str) -> bool:
        """Interactive yes/no prompt for the over-capacity scenario."""
        prompt_str = t("capacity_confirm_prompt", language)
        yes_key = t("capacity_confirm_yes", language)
        try:
            if self.prompt_mode == "simple":
                answer = input(prompt_str).strip().lower()
            else:
                if self._prompt_session is None:
                    self._prompt_session = self._make_prompt_session()
                answer = self._prompt_session.prompt(prompt_str).strip().lower()
            return answer == yes_key
        except Exception:
            return False

    def _ask_feedback(
        self,
        agent_name: str,
        output_text: str,
        scenario_type: str,
        query: str,
        key_facts: dict,
    ) -> None:
        """Collect rating + optional correction for an agent output and persist it."""
        if not self.feedback_enabled:
            return
        rating_prompt = t("feedback_prompt", self.language)
        correction_prompt = t("feedback_correction_prompt", self.language)
        try:
            raw = input(rating_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if raw not in ("1", "2", "3"):
            return
        rating = int(raw)
        try:
            correction = input(correction_prompt).strip() or None
        except (EOFError, KeyboardInterrupt):
            correction = None
        if rating == 1 and not correction:
            return
        save_feedback(
            agent=agent_name,
            lang=self.language,
            rating=rating,
            scenario_type=scenario_type,
            query=query,
            key_facts=key_facts,
            output=output_text,
            correction=correction,
        )
        print_message(t("feedback_saved", self.language), style=MSG_OK)

    def _handle_query(
        self,
        query: str,
        _prebuilt: Optional[tuple[MILPResult, ScenarioParams]] = None,
    ) -> None:
        """Run the full pipeline for a user query."""
        self.query_count += 1
        qnum = self.query_count

        spinner = _TimedSpinner(t("processing", self.language))

        result: Optional[PipelineResult] = None
        over_capacity = False

        if _prebuilt is not None:
            # Bypass classifier and OR agent — params are already known (e.g. baseline)
            milp_result_raw, scenario_params = _prebuilt
        else:
            # ── Intent classification ─────────────────────────────────────────────
            with Live(spinner, console=console, refresh_per_second=12, transient=True):
                intent = classify_intent(query, self.language, self.classifier_agent)

            if intent == "shock_response":
                self._handle_shock_response(query)
                return

            # ── Existing what-if pipeline ─────────────────────────────────────────
            # Phase 1: OR Agent → scenario params
            with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                try:
                    milp_result_raw, scenario_params = run_or_agent(self.or_agent, query)
                except Exception as exc:
                    live.stop()
                    _print_pipeline_error(exc, self.language)
                    return

        # Baseline guard: prompt user when a non-baseline scenario arrives without a reference
        if self.baseline_result is None and not scenario_params.is_baseline:
            if self._ask_baseline_confirmation(self.language):
                self._handle_baseline()
            # continue regardless — user either ran baseline or chose to skip

        # Phase 2: Capacity gate — check before running models (no spinner, interactive)
        if (
            scenario_params.terminal_demand_multipliers
            and any(v > 1.0 for v in scenario_params.terminal_demand_multipliers.values())
        ):
            cap_check = check_capacity(self.network, scenario_params)
            if cap_check.has_overflow:
                print_capacity_warning(cap_check, self.network, self.language)
                confirmed = self._ask_over_capacity_confirmation(self.language)
                if not confirmed:
                    return
                scenario_params.skip_capacity_constraints = True
                over_capacity = True

        # Phase 3: Models + Expert
        with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
            try:
                result = run_pipeline_from_params(
                    milp_result=milp_result_raw,
                    scenario_params=scenario_params,
                    network=self.network,
                    expert_agent=self.expert_agent,
                    query_number=qnum,
                    language=self.language,
                    llm_insights=self.llm_insights,
                    baseline_result=self.baseline_result,
                )
            except Exception as exc:
                live.stop()
                _print_pipeline_error(exc, self.language)
                return

        if result is None:
            return

        result.query_text = query
        self.last_result = result
        self.scenario_history.append(result)

        if result.scenario_params.is_baseline:
            self.baseline_result = result

        if not result.milp_result.feasible:
            print_infeasible_explanation(result.milp_result, result.scenario_params, self.language)
            console.print()
        elif result.insight:
            print_insight(result.insight, self.language)
            console.print()
            self._ask_feedback(
                agent_name="transportation_expert",
                output_text=result.insight,
                scenario_type=detect_scenario_type(result.scenario_params, result.milp_result),
                query=query,
                key_facts={
                    "trucks": result.milp_result.trucks,
                    "cost": result.milp_result.total_cost,
                    "served_cps": len(result.milp_result.served_cps),
                    "total_cps": len(self.network.cp_ids),
                },
            )

        if self.llm_insights and not self._ask_show_tables():
            return

        if not result.scenario_params.is_baseline and self.baseline_result is not None:
            print_param_recap(
                result.scenario_params,
                self.baseline_result.scenario_params,
                self.language,
            )

        baseline = None if result.scenario_params.is_baseline else self.baseline_result
        print_summary_table(result, baseline, self.language)
        console.print()

        print_served_cps(result.milp_result, self.network.cp_ids, self.language)

        p = result.scenario_params
        has_closure   = any(not v for v in p.terminals_active.values())
        has_demand    = bool(p.terminal_demand_multipliers)
        has_partial   = result.milp_result.coverage_count < len(self.network.cp_ids)
        has_vol_cap   = bool(p.terminal_volume_caps)
        if (
            result.milp_result.feasible
            and result.milp_result.volumes
            and (
                (not p.volume_redistribution and (has_closure or has_demand or has_partial))
                or (p.volume_redistribution and has_vol_cap)
            )
        ):
            is_pt = self.language == "pt"
            if p.volume_redistribution and has_vol_cap:
                vol_title = (
                    "Distribuição de Volume — Redistribuição Ótima" if is_pt
                    else "Volume Distribution — Optimal Redistribution"
                )
            elif has_closure:
                vol_title = None  # default: "Distribuição de Volume — Terminais Ativos"
            elif has_demand:
                vol_title = (
                    "Demanda e Capacidade — Cenário Ajustado" if is_pt
                    else "Demand & Capacity — Adjusted Scenario"
                )
            else:
                vol_title = (
                    "Distribuição de Volume — Cobertura Parcial" if is_pt
                    else "Volume Distribution — Partial Coverage"
                )
            print_volume_matrix(
                result.milp_result,
                self.network,
                result.scenario_params.terminals_active,
                self.language,
                title=vol_title,
            )

        if over_capacity and result.milp_result.terminal_overflows:
            print_over_capacity_highlights(result.milp_result, self.network, self.language)
            console.print()

        if over_capacity and result.milp_result.cp_overflows:
            print_cp_over_capacity_highlights(result.milp_result, self.network, self.language)
            console.print()

    def _handle_shock_response(self, query: str) -> None:
        """Run the shock response agent for adversarial/compensation queries."""
        spinner = _TimedSpinner(t("processing", self.language))

        session_context = ""
        if self.scenario_history:
            profile = classify_session(self.scenario_history)
            session_context = serialize_scenarios(
                self.scenario_history, self.network, profile, self.language
            )

        with Live(spinner, console=console, refresh_per_second=12, transient=True):
            try:
                provider, _ = MODEL_REGISTRY[self.model_alias]
                agentic_model = get_agent_model(provider, "agentic")
                agent = create_shock_response_agent(provider, agentic_model, self.language)
                output = run_shock_response_agent(agent, query, session_context, self.language)
            except Exception as exc:
                _print_pipeline_error(exc, self.language)
                return

        render_shock_response(output, self.language)
        # NOTE: output is intentionally NOT appended to self.scenario_history.
        self._ask_feedback(
            agent_name="shock_response",
            output_text=output.narrative,
            scenario_type="shock_response",
            query=query,
            key_facts={
                "shock_description": output.shock_description,
                "strategies_count": len(output.strategies),
                "winning_strategy": output.strategies[0].strategy_name if output.strategies else "",
                "cost_recovered": output.strategies[0].cost_recovered if output.strategies else 0,
            },
        )

    def _handle_relocate(self) -> None:
        """Run the volume redistribution scenario directly."""
        self.query_count += 1
        qnum = self.query_count

        base_params = (
            self.baseline_result.scenario_params
            if self.baseline_result is not None
            else build_baseline_params(self.network)
        )

        result: Optional[PipelineResult] = None
        with Live(
            _TimedSpinner(t("processing", self.language)),
            console=console,
            refresh_per_second=12,
            transient=True,
        ) as live:
            try:
                result = run_relocation_pipeline(
                    network=self.network,
                    expert_agent=self.expert_agent,
                    base_params=base_params,
                    query_number=qnum,
                    language=self.language,
                    llm_insights=self.llm_insights,
                    baseline_result=self.baseline_result,
                )
            except Exception as exc:
                live.stop()
                _print_pipeline_error(exc, self.language)
                return

        if result is None:
            return

        result.query_text = "/relocate" if self.language == "en" else "/realocar"
        self.last_result = result
        self.scenario_history.append(result)

        if not result.milp_result.feasible:
            print_infeasible_explanation(result.milp_result, result.scenario_params, self.language)
            console.print()
        elif result.insight:
            print_insight(result.insight, self.language)
            console.print()

        if self.llm_insights and not self._ask_show_tables():
            return

        print_param_recap(
            result.scenario_params,
            base_params,
            self.language,
        )
        print_summary_table(result, self.baseline_result, self.language)
        console.print()

    def _handle_network_relocate(self) -> None:
        """Show the CP→terminal relocation map from the last redistribution result."""
        result = self.last_result
        if result is None or not result.scenario_params.volume_redistribution:
            print_message(t("no_relocate_result", self.language), style=MSG_QUIET)
            return
        print_network_relocate(result, self.network, self.language)
        console.print()

    def _handle_baseline(self) -> None:
        """Run or re-display the baseline scenario."""
        if self.baseline_result is not None:
            print_message(t("baseline_reuse", self.language), style=MSG_INFO)
            console.print()
            if self.baseline_result.insight:
                print_insight(self.baseline_result.insight, self.language)
                console.print()
            if self.llm_insights and not self._ask_show_tables():
                return
            print_summary_table(self.baseline_result, None, self.language)
            console.print()
            return

        print_message(t("baseline_running", self.language), style=MSG_INFO)

        # Bypass the OR agent: baseline params are fully deterministic.
        # Run the MILP solver directly, then hand off to _handle_query with _prebuilt
        # to skip the classifier and OR agent API calls entirely.
        baseline_params = build_baseline_params(self.network)
        with Live(
            _TimedSpinner(t("processing", self.language)),
            console=console,
            refresh_per_second=12,
            transient=True,
        ) as live:
            try:
                baseline_milp = run_milp_solver(self.network, baseline_params)
            except Exception as exc:
                live.stop()
                _print_pipeline_error(exc, self.language)
                return

        baseline_query = (
            "Qual é o cenário baseline considerando todos os requisitos atuais?"
            if self.language == "pt"
            else "What is the baseline scenario considering all as-is requirements?"
        )
        self._handle_query(baseline_query, _prebuilt=(baseline_milp, baseline_params))
        if self.last_result is not None:
            print_message(t("baseline_done", self.language), style=MSG_INFO)

    def _handle_lane_costs(self) -> None:
        """Show operating cost ranking by MILP-assigned lane, lowest to highest $/ton."""
        if self.baseline_result is not None:
            milp = self.baseline_result.milp_result
        else:
            print_message(t("baseline_running", self.language), style=MSG_INFO)
            with Live(
                _TimedSpinner(t("processing", self.language)),
                console=console,
                refresh_per_second=12,
                transient=True,
            ) as live:
                try:
                    params = build_baseline_params(self.network)
                    milp = run_milp_solver(self.network, params)
                except Exception as exc:
                    live.stop()
                    _print_pipeline_error(exc, self.language)
                    return
            if not milp.feasible:
                print_message(
                    "Baseline solver returned no feasible solution." if self.language == "en"
                    else "O solver baseline não encontrou solução viável.",
                    style=MSG_ERR,
                )
                return
        handle_lane_costs(self.network, milp, self.language)

    def _handle_detail(self, args: str) -> None:
        """Show detail metrics for a model."""
        if self.last_result is None:
            print_message(t("detail_not_available", self.language), style=MSG_QUIET)
            return

        model_key = args.lower().strip()
        if model_key in ("lane-by-lane", "lane_by_lane", "lbl", "lane"):
            print_detail(
                t("row_lbl", self.language),
                self.last_result.lbl_result,
                self.language,
            )
        elif model_key in ("weighted", "wct", "weighted-cycle", "weighted_cycle"):
            print_detail(
                t("row_wct", self.language),
                self.last_result.wct_result,
                self.language,
            )
        elif model_key in ("solver", "milp", "otimizador"):
            print_detail(
                t("row_milp", self.language),
                self.last_result.milp_result,
                self.language,
                milp=True,
                total_cps=len(self.network.cp_ids),
            )
        else:
            print_message(
                "Usage: /detail lane-by-lane | weighted | solver",
                style=MSG_QUIET,
            )

    def _handle_export(self) -> None:
        """Export the last result to xlsx."""
        if self.last_result is None:
            print_message(t("no_result", self.language), style=MSG_QUIET)
            return
        path = export_result(self.last_result, self.last_result.query_number)
        print_message(t("export_saved", self.language, path=str(path)), style=MSG_OK)

    def _handle_model(self, args: str) -> None:
        """Switch the active AI provider (anthropic or openai)."""
        alias = args.lower().strip()
        if alias not in ("anthropic", "openai"):
            print_message(
                t("model_usage", self.language, current=self.model_alias),
                style=MSG_QUIET,
            )
            return
        provider = alias
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print_message(
                "OPENAI_API_KEY not set. Export it before switching to OpenAI.",
                style=MSG_ERR,
            )
            return
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print_message(
                "ANTHROPIC_API_KEY not set. Export it before switching to Anthropic.",
                style=MSG_ERR,
            )
            return
        self.model_alias = alias
        self._init_agents()
        print_message(
            t("model_switched", self.language, provider=alias),
            style=MSG_INFO,
        )

    def _handle_learning(self, args: str) -> None:
        """Toggle the feedback learning loop on or off."""
        flag = args.lower().strip()
        if flag == "on":
            self.feedback_enabled = True
            print_message(t("learning_on", self.language), style=MSG_OK)
        elif flag == "off":
            self.feedback_enabled = False
            print_message(t("learning_off", self.language), style=MSG_WARN)
        else:
            state = t("learning_state_on", self.language) if self.feedback_enabled else t("learning_state_off", self.language)
            print_message(t("learning_usage", self.language, state=state), style=MSG_QUIET)

    def _handle_llm(self, args: str) -> None:
        """Toggle LLM insight generation on or off."""
        flag = args.lower().strip()
        if flag == "on":
            self.llm_insights = True
            print_message(t("llm_on", self.language), style=MSG_OK)
        elif flag == "off":
            self.llm_insights = False
            print_message(t("llm_off", self.language), style=MSG_WARN)
        else:
            state = t("llm_state_on", self.language) if self.llm_insights else t("llm_state_off", self.language)
            print_message(
                t("llm_usage", self.language, state=state),
                style=MSG_QUIET,
            )

    def _cmd_data_expert(self) -> None:
        """Cross-scenario analysis of the full session history."""
        if len(self.scenario_history) < 2:
            print_message(t("data_expert_need_scenarios", self.language), style=MSG_QUIET)
            return
        spinner = _TimedSpinner(t("data_expert_running", self.language))
        with Live(spinner, console=console, refresh_per_second=12, transient=True):
            try:
                profile = classify_session(self.scenario_history)
                serialized = serialize_scenarios(
                    self.scenario_history, self.network, profile, self.language
                )
                output = run_data_expert_agent(
                    serialized,
                    profile,
                    self.network.terminal_ids,
                    self.language,
                    self.model_alias,
                    self.scenario_history,
                    self.network,
                )
            except Exception as exc:
                _print_pipeline_error(exc, self.language)
                return
        render_data_expert(
            output,
            profile,
            self.network.terminal_ids,
            self.language,
            total_cps=len(self.network.cp_ids),
        )
        self._ask_feedback(
            agent_name="data_expert",
            output_text=output.narrative,
            scenario_type="session_analysis",
            query="/data-expert",
            key_facts={"scenario_count": len(self.scenario_history)},
        )

    def _handle_language(self, args: str) -> None:
        """Switch the active language."""
        lang = args.lower().strip()
        if lang not in ("pt", "en"):
            print_message("Available languages: pt, en", style=MSG_ERR)
            return
        self.language = lang
        self._init_agents()
        print_message(t("language_switched", lang), style=MSG_INFO)


def run() -> None:
    """Entry point for the CLI."""
    cli = FleetPlanningCLI()
    cli.run()
