"""Agents network topology classes."""
import networkx as nx
from bokeh.models import (
    Plot,
    Range1d,
    MultiLine,
    Circle,
    HoverTool,
    TapTool,
    )
from bokeh.models.graphs import (
    from_networkx,
    NodesOnly,
    NodesAndLinkedEdges,
    )
from bokeh.palettes import Spectral4


class StaticAgentsNetwork():
    """Time-invariant (static) agents network."""

    def __init__(self, model, graph):
        """Initialize agents network based on a graph.

        Parameters
        ----------
        model : Mesa model
            model which contains agents that should be arranged into a network.

        graph : NetworkX graph object
            graph that describes the network topology

        """
        self.graph = graph
        self.model = model
        self.curr_idx = 0

    def add_agent(self, agent):
        """Add an agent to the network.

        HACK: This whole function is a hack. Basically it has a counter which
        starts at zero when an object is created. Then it counts up with each
        added agent, so one can add agent only once!

        Parameters
        ----------
        agent : Mesa Agent object
            agent to be added to the graph.

        """
        if self.curr_idx == len(self.graph.nodes):
            raise ValueError("Cannot relabel any more nodes")
        mapping = {self.curr_idx: agent.unique_id}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        self.curr_idx += 1

    def get_neighbours(self, agent):
        """Get agent's neighbours.

        Parameters
        ----------
        agent : Mesa Agent object
            agent whose neighbours will be returned.

        """
        return filter(
            lambda a: a.unique_id in self.graph.neighbors(agent.unique_id),
            self.model.schedule.agents,
            )

    def get_bokeh_plot(self):
        """Get a nice Bokeh plot of the network."""
        plot = Plot(
            plot_width=400,
            plot_height=400,
            x_range=Range1d(-1.1, 1.1),
            y_range=Range1d(-1.1, 1.1),
            )
        plot.title.text = "Agents network"

        plot.add_tools(
            HoverTool(tooltips=[("AgentID", "@index")]),
            TapTool(),
            )

        graph_renderer = from_networkx(
            self.graph,
            nx.spring_layout,
            scale=1,
            center=(0, 0),
            )

        graph_renderer.node_renderer.glyph = Circle(
            size=15,
            fill_color=Spectral4[0],
            )
        graph_renderer.node_renderer.selection_glyph = Circle(
            size=15,
            fill_color=Spectral4[2],
            )
        graph_renderer.node_renderer.hover_glyph = Circle(
            size=15,
            fill_color=Spectral4[1],
            )

        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#CCCCCC",
            line_alpha=0.8,
            line_width=3,
            )
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color=Spectral4[2],
            line_width=3,
            )

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesOnly()

        plot.renderers.append(graph_renderer)
        return plot
