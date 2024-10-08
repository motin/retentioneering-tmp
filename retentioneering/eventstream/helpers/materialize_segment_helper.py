from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.eventstream.types import AddSegmentType, EventstreamType
from retentioneering.utils.doc_substitution import docstrings


class MaterializeSegmentHelperMixin:
    @docstrings.with_indent(12)
    @time_performance(
        scope="add_start_end_events",
        event_name="helper",
        event_value="combine",
    )
    def materialize_segment(self, name: str) -> EventstreamType:
        """
        Propagate segment synthetic events to a column.

        Parameters
        ----------
            %(MaterializeSegment.parameters)s

        Returns
        -------
        EventstreamType
            Eventstream with added column.
        """
        from retentioneering.data_processors_lib import (
            MaterializeSegment,
            MaterializeSegmentParams,
        )
        from retentioneering.preprocessing_graph import PreprocessingGraph
        from retentioneering.preprocessing_graph.nodes import EventsNode

        p = PreprocessingGraph(source_stream=self)  # type: ignore
        node = EventsNode(processor=MaterializeSegment(params=MaterializeSegmentParams(name=name)))  # type: ignore
        p.add_node(node=node, parents=[p.root])
        result = p.combine(node)
        del p
        # todo @dakhaytin - fill params
        # collect_data_performance(
        #     scope="add_start_end_events",
        #     event_name="metadata",
        #     called_params={},
        #     performance_data={},
        #     eventstream_index=self._eventstream_index,
        #     parent_eventstream_index=self._eventstream_index,
        #     child_eventstream_index=result._eventstream_index,
        # )

        return result
