from .base import BaseGenerator, ParamSpec
from .bar_chart_value import BarChartValueGenerator
from .bar_height import BarHeightGenerator
from .flowchart import FlowchartGenerator
from .grid_lookup import GridLookupGenerator
from .grid_retrieval import GridRetrievalGenerator
from .grid_value_search import GridValueSearchGenerator
from .line_chart_legend_mapping import LineChartLegendMappingGenerator
from .line_chart import LineChartGenerator
from .line_intersection import LineIntersectionGenerator
from .path_following import PathFollowingGenerator
from .pie_chart import PieChartGenerator
from .region_containment import RegionContainmentGenerator
from .scatter_color_count import ScatterColorCountGenerator
from .scatter_plot_legend_mapping import ScatterPlotLegendMappingGenerator
from .scatter_region_count import ScatterRegionCountGenerator
from .scatter_shape_count import ScatterShapeCountGenerator
from .swimlane import SwimlaneGenerator

# Registry of available generators
GENERATORS = {
    "bar_chart_value": BarChartValueGenerator,
    "bar_height": BarHeightGenerator,
    "flowchart": FlowchartGenerator,
    "grid_lookup": GridLookupGenerator,
    "grid_retrieval": GridRetrievalGenerator,
    "grid_value_search": GridValueSearchGenerator,
    "line_chart_legend_mapping": LineChartLegendMappingGenerator,
    "line_chart": LineChartGenerator,
    "line_intersection": LineIntersectionGenerator,
    "path_following": PathFollowingGenerator,
    "pie_chart": PieChartGenerator,
    "region_containment": RegionContainmentGenerator,
    "scatter_color_count": ScatterColorCountGenerator,
    "scatter_plot_legend_mapping": ScatterPlotLegendMappingGenerator,
    "scatter_region_count": ScatterRegionCountGenerator,
    "scatter_shape_count": ScatterShapeCountGenerator,
    "swimlane": SwimlaneGenerator,
}

__all__ = [
    "BaseGenerator",
    "ParamSpec",
    "GENERATORS",
]
