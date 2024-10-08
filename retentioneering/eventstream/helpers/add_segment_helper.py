from typing import Optional

from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.eventstream.types import AddSegmentType, EventstreamType
from retentioneering.utils.doc_substitution import docstrings


class AddSegmentHelperMixin:
    @docstrings.with_indent(12)
    @time_performance(
        scope="add_start_end_events",
        event_name="helper",
        event_value="combine",
    )
    def add_segment(self, segment: AddSegmentType, name: Optional[str] = None) -> EventstreamType:
        """
        Add segment synthetic events to eventstream.

        Parameters
        ----------
            %(AddSegment.parameters)s

        Returns
        -------
        EventstreamType
            Eventstream with added segment.
        """

        from retentioneering.data_processors_lib import AddSegment, AddSegmentParams
        from retentioneering.preprocessing_graph import PreprocessingGraph
        from retentioneering.preprocessing_graph.nodes import EventsNode

        p = PreprocessingGraph(source_stream=self)  # type: ignore
        node = EventsNode(processor=AddSegment(params=AddSegmentParams(segment=segment, name=name)))  # type: ignore
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
