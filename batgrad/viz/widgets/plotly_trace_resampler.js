function axesFromLayout(layout) {
  const axes = {};
  for (const [name, value] of Object.entries(layout || {})) {
    if (!/^[xy]axis/.test(name) || value?.autorange === true || !Array.isArray(value?.range)) continue;
    axes[name] = {
      range: value.range.slice(),
      autorange: value.autorange === true,
      type: value.type || "linear",
    };
  }
  return axes;
}

function visibleTraces(data) {
  const visible = {};
  for (let idx = 0; idx < (data || []).length; idx += 1) {
    const trace = data[idx] || {};
    visible[idx] = !(trace.visible === false || trace.visible === "legendonly");
  }
  return visible;
}

function numericRange(values) {
  const numeric = values.map(Number).filter(Number.isFinite);
  if (!numeric.length) return null;
  return [Math.min(...numeric), Math.max(...numeric)];
}

function axisKey(axis) {
  if (!axis || axis === "x" || axis === "y") return axis || "";
  return axis.replace("axis", "");
}

function selectionAxisRange(evt, axis) {
  const key = axisKey(axis);
  const boxRange = evt?.range?.[key];
  if (Array.isArray(boxRange) && boxRange.length >= 2) return numericRange(boxRange);
  const lassoRange = evt?.lassoPoints?.[key];
  if (Array.isArray(lassoRange) && lassoRange.length >= 2) return numericRange(lassoRange);
  return null;
}

function selectionFromEvent(evt, data) {
  const byTrace = new Map();
  for (const point of evt?.points || []) {
    const traceIdx = Number.isInteger(point?.curveNumber) ? point.curveNumber : null;
    if (traceIdx === null) continue;
    const current = byTrace.get(traceIdx) || { x: [], y: [], count: 0 };
    current.x.push(point.x);
    current.y.push(point.y);
    current.count += 1;
    byTrace.set(traceIdx, current);
  }
  const traces = [];
  for (const [traceIdx, selected] of byTrace.entries()) {
    const trace = data?.[traceIdx] || {};
    const xaxis = trace.xaxis || "x";
    const yaxis = trace.yaxis || "y";
    const xRange = selectionAxisRange(evt, xaxis);
    const yRange = selectionAxisRange(evt, yaxis);
    if (!xRange || !yRange) continue;
    traces.push({
      trace_idx: traceIdx,
      name: trace.name || `${traceIdx}`,
      legendgroup: trace.legendgroup || "",
      xaxis,
      yaxis,
      x_range: xRange,
      y_range: yRange,
      point_count: selected.count,
    });
  }
  traces.sort((left, right) => left.trace_idx - right.trace_idx);
  return { traces };
}

function mergeSelection(previous, next) {
  const byTrace = new Map();
  for (const selected of previous?.traces || []) {
    if (!Number.isInteger(selected?.trace_idx)) continue;
    byTrace.set(selected.trace_idx, selected);
  }
  for (const selected of next?.traces || []) {
    if (!Number.isInteger(selected?.trace_idx)) continue;
    byTrace.set(selected.trace_idx, selected);
  }
  const traces = [...byTrace.values()].sort((left, right) => left.trace_idx - right.trace_idx);
  return { traces };
}

export default {
  async render({ model, el }) {
    const loadPlotly = () => {
      if (window.Plotly) return Promise.resolve(window.Plotly);
      if (!window.__batgradPlotlyPromise) {
        window.__batgradPlotlyPromise = new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = "https://cdn.plot.ly/plotly-3.6.0.min.js";
          script.onload = () => resolve(window.Plotly);
          script.onerror = reject;
          document.head.appendChild(script);
        });
      }
      return window.__batgradPlotlyPromise;
    };

    const Plotly = await loadPlotly();
    const fig = model.get("_fig_json") || { data: [], layout: {} };

    const frame = document.createElement("div");
    frame.className = "batgrad-resampler-frame";
    const plot = document.createElement("div");
    plot.className = "batgrad-resampler-plot";
    plot.style.height = `${model.get("_height") || 600}px`;
    const status = document.createElement("div");
    status.className = "batgrad-resampler-status";
    status.textContent = model.get("_status") || "Preparing...";
    frame.append(plot, status);
    el.replaceChildren(frame);

    await Plotly.newPlot(plot, fig.data || [], fig.layout || {}, {
      responsive: true,
      scrollZoom: true,
      displayModeBar: true,
    });

    let requestId = 0;
    let latestAppliedRequestId = 0;
    let relayoutTimer = null;
    let suppressRelayout = false;
    let suppressRestyle = false;

    const emitViewport = () => {
      requestId += 1;
      status.textContent = "Sampling...";
      model.set("_evt", {
        axes: axesFromLayout(plot.layout),
        visible: visibleTraces(plot.data),
        _rid: requestId,
      });
      model.save_changes();
    };

    const setTraceVisibility = async visible => {
      const indices = (plot.data || []).map((_trace, idx) => idx);
      suppressRestyle = true;
      try {
        await Plotly.restyle(plot, { visible }, indices);
      } finally {
        suppressRestyle = false;
      }
      emitViewport();
    };

    plot.on("plotly_relayout", evt => {
      if (suppressRelayout || !evt) return;
      const keys = Object.keys(evt);
      const hasRange = keys.some(key => key.includes(".range") || key.includes(".autorange"));
      if (!hasRange) return;
      clearTimeout(relayoutTimer);
      relayoutTimer = setTimeout(emitViewport, 80);
    });

    plot.on("plotly_click", async evt => {
      const point = Array.isArray(evt?.points) ? evt.points[0] : null;
      const traceIdx = Number.isInteger(point?.curveNumber) ? point.curveNumber : null;
      if (traceIdx === null) return;
      const group = plot.data?.[traceIdx]?.legendgroup;
      if (group === undefined || group === null || group === "") return;
      const visible = (plot.data || []).map(trace =>
        trace?.legendgroup === group ? true : "legendonly",
      );
      await setTraceVisibility(visible);
    });

    plot.on("plotly_restyle", evt => {
      if (suppressRestyle) return;
      const update = Array.isArray(evt) ? evt[0] : evt;
      if (update && Object.prototype.hasOwnProperty.call(update, "visible")) {
        emitViewport();
      }
    });

    plot.on("plotly_selected", evt => {
      model.set(
        "selection",
        mergeSelection(model.get("selection"), selectionFromEvent(evt, plot.data)),
      );
      model.save_changes();
    });

    plot.on("plotly_deselect", () => {
      model.set("selection", {});
      model.save_changes();
    });

    const applyUpdate = update => {
      if (!update) return;
      const updateRid = Number(update._rid || 0);
      if (updateRid < latestAppliedRequestId) return;
      latestAppliedRequestId = updateRid;
      status.textContent = update.status || "Ready";
      if (!Array.isArray(update.updates)) return;
      for (const traceUpdate of update.updates) {
        const hasPoints = Array.isArray(traceUpdate.x) && traceUpdate.x.length > 0;
        const data = hasPoints
          ? { x: [traceUpdate.x || []], y: [traceUpdate.y || []] }
          : { x: [[null]], y: [[null]] };
        if (hasPoints && Object.prototype.hasOwnProperty.call(traceUpdate, "customdata")) {
          data.customdata = [traceUpdate.customdata || []];
        }
        Plotly.restyle(plot, data, [traceUpdate.trace_idx]);
      }
    };

    const updateListener = () => applyUpdate(model.get("_update"));
    model.on("change:_update", updateListener);
    applyUpdate(model.get("_update"));

    return () => {
      clearTimeout(relayoutTimer);
      if (typeof model.off === "function") model.off("change:_update", updateListener);
      try {
        Plotly.purge(plot);
      } catch {
        // no-op
      }
      el.replaceChildren();
      suppressRelayout = true;
    };
  },
};
