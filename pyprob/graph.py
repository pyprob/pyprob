import pydotplus
import os
import math
import numpy as np
from collections import OrderedDict
from itertools import islice

from . import util, diagnostics
from .distributions import Empirical


class Node():
    def __init__(self, address_id, variable, weight):
        self.address_id = address_id
        self.variable = variable
        self.weight = weight
        self.outgoing_edges = []
        if variable is None:
            self.color = '#ffffff'
        else:
            if variable.control:
                if variable.replace:
                    self.color = '#adff2f'
                else:
                    self.color = '#fa8072'
            elif variable.observable:
                self.color = '#1effff'
                if variable.observed:
                    self.color = '#1e90ff'
            elif variable.tagged:
                self.color = '#cccccc'
            else:
                self.color = '#ffd700'

    def add_outgoing_edge(self, node, weight):
        edge = Edge(self, node, weight)
        self.outgoing_edges.append(edge)
        return edge

    def __repr__(self):
        return 'Node(address_id:{}, weight:{}, outgoing_edges:{})'.format(self.address_id, self.weight, [str(edge) for edge in self.outgoing_edges])


class Edge():
    def __init__(self, node_0, node_1, weight):
        self.node_0 = node_0
        self.node_1 = node_1
        self.weight = weight

    def __repr__(self):
        return 'Edge(node_0: {}, node_1:{}, weight:{})'.format(self.node_0.address_id, self.node_1.address_id, self.weight)


class Graph():
    def __init__(self, trace_dist, base_graph=None, use_address_base=True, n_most_frequent=None, normalize_weights=True):
        self.nodes = []
        self.edges = []
        if base_graph is None:
            self.address_stats = None
            self.trace_stats = None
            self.use_address_base = use_address_base
        else:
            self.address_stats = base_graph.address_stats
            self.trace_stats = base_graph.trace_stats
            self.use_address_base = base_graph.use_address_base

        # self.address_stats = diagnostics._address_stats(trace_dist, use_address_base=self.use_address_base)

        self.trace_stats = diagnostics._trace_stats(trace_dist, use_address_base=self.use_address_base, reuse_ids_from_address_stats=self.address_stats, reuse_ids_from_trace_stats=self.trace_stats)
        self.address_stats = self.trace_stats['address_stats']
        address_id_to_variable = self.address_stats['address_id_to_variable']
        traces = self.trace_stats['traces']
        traces = OrderedDict(sorted(dict(traces).items(), key=lambda x: x[1]['count'], reverse=True))
        if n_most_frequent is not None:
            # n_most_frequent = len(self.trace_stats)
            traces = dict(islice(traces.items(), n_most_frequent))

        nodes = {}
        edges = {}
        for key, value in traces.items():
            weight = value['weight']
            address_id_sequence = value['address_id_sequence']
            for address_id in address_id_sequence:
                if address_id in nodes:
                    nodes[address_id] += weight
                else:
                    nodes[address_id] = weight
            for left, right in zip(address_id_sequence, address_id_sequence[1:]):
                if (left, right) in edges:
                    edges[(left, right)] += weight
                else:
                    edges[(left, right)] = weight

        for edge, weight in edges.items():
            address_id_0 = edge[0]
            node_0 = self.get_node(address_id_0)
            if node_0 is None:
                if address_id_0 in address_id_to_variable:
                    variable_0 = address_id_to_variable[address_id_0]
                else:
                    variable_0 = None
                node_0 = Node(address_id_0, variable_0, nodes[address_id_0])
                self.add_node(node_0)

            address_id_1 = edge[1]
            node_1 = self.get_node(address_id_1)
            if node_1 is None:
                if address_id_1 in address_id_to_variable:
                    variable_1 = address_id_to_variable[address_id_1]
                else:
                    variable_1 = None
                node_1 = Node(address_id_1, variable_1, nodes[address_id_1])
                self.add_node(node_1)

            self.add_edge(node_0.add_outgoing_edge(node_1, weight))

        if normalize_weights:
            self.normalize_weights()

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, address_id):
        return next((node for node in self.nodes if node.address_id == address_id), None)

    def add_edge(self, edge):
        self.edges.append(edge)

    def normalize_weights(self):
        node_weight_total = 0
        for node in self.nodes:
            node_weight_total += node.weight
            edge_weight_total = 0
            for edge in node.outgoing_edges:
                edge_weight_total += edge.weight
            if edge_weight_total > 0:
                for edge in node.outgoing_edges:
                    edge.weight /= edge_weight_total

        for node in self.nodes:
            node.weight /= node_weight_total

    def trace_graphs(self):
        traces = self.trace_stats['traces']
        for key, val in traces.items():
            trace = val['trace']
            trace_id = val['trace_id']
            yield trace_id, Graph(Empirical([trace]), base_graph=self, use_address_base=self.use_address_base)

    def render_to_graphviz(self, background_graph=None):
        if background_graph is None:
            graph = pydotplus.graphviz.Dot(graph_type='digraph', rankdir='LR')
        else:
            graph = pydotplus.graphviz.graph_from_dot_data(background_graph.render_to_graphviz())
            for node in graph.get_nodes():
                node.set_color('#cccccc')
                node.set_fontcolor('#cccccc')
            for edge in graph.get_edges():
                edge.set_color('#cccccc')
                edge.set_fontcolor('#cccccc')
                edge.set_label(' ')

        for edge in self.edges:
            node_0 = edge.node_0
            nodes = graph.get_node(node_0.address_id)
            if len(nodes) > 0:
                graph_node_0 = nodes[0]
            else:
                graph_node_0 = pydotplus.Node(node_0.address_id)
                graph.add_node(graph_node_0)
            graph_node_0.set_style('filled')
            graph_node_0.set_fillcolor(node_0.color)
            graph_node_0.set_color('black')
            graph_node_0.set_fontcolor('black')
            color_factor = 0.75 * (math.exp(1. - node_0.weight) - 1.) / (math.e - 1.)
            graph_node_0.set_penwidth(max(0.1, 4 * (1 - color_factor)))

            node_1 = edge.node_1
            nodes = graph.get_node(node_1.address_id)
            if len(nodes) > 0:
                graph_node_1 = nodes[0]
            else:
                graph_node_1 = pydotplus.Node(node_1.address_id)
                graph.add_node(graph_node_1)
            graph_node_1.set_style('filled')
            graph_node_1.set_fillcolor(node_1.color)
            graph_node_1.set_color('black')
            graph_node_1.set_fontcolor('black')
            color_factor = 0.75 * (math.exp(1. - node_1.weight) - 1.) / (math.e - 1.)
            graph_node_1.set_penwidth(max(0.25, 5 * (1 - color_factor)))

            edges = graph.get_edge(node_0.address_id, node_1.address_id)
            if len(edges) > 0:
                graph_edge = edges[0]
            else:
                graph_edge = pydotplus.Edge(graph_node_0, graph_node_1, weight=max(edge.weight, 1e-3))  # pydotplus fails with extremely small weights
                graph.add_edge(graph_edge)
            # if background_graph is None:
            graph_edge.set_label('\"{:,.3f}\"'.format(edge.weight))
            color_factor = 0.75 * (math.exp(1. - edge.weight) - 1.) / (math.e - 1.)
            graph_edge.set_color(util.rgb_to_hex((color_factor, color_factor, color_factor)))
            graph_edge.set_fontcolor('black')
            # else:
                # graph_edge.set_color('black')
                # graph_edge.set_fontcolor('black')

        return graph.to_string()

    def render_to_file(self, file_name, background_graph=None):
        graph = self.render_to_graphviz(background_graph)
        file_name_dot = file_name + '.dot'
        with open(file_name_dot, 'w') as file:
            file.write(graph)
        file_name_pdf = file_name + '.pdf'
        status = os.system('dot -Tpdf {} -o {}'.format(file_name_dot, file_name_pdf))
        if status != 0:
            print('Cannot not render to file {}. Check that GraphViz is installed.'.format(file_name_pdf))

    def sample_execution(self):
        node = self.get_node('START')
        seq = [node]
        while node.address_id != 'END':
            weights = [edge.weight for edge in node.outgoing_edges]
            edge = np.random.choice(node.outgoing_edges, 1, p=weights)[0]
            node = edge.node_1
            seq.append(node)
        return seq
