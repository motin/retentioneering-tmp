from retentioneering.backend.tracker import time_performance
from retentioneering.eventstream.types import EventstreamType
from retentioneering.utils.doc_substitution import docstrings


class DropSegmentHelperMixin:
    @docstrings.with_indent(12)
    @time_performance(
        scope="add_start_end_events",
        event_name="helper",
        event_value="combine",
    )
    def drop_segment(self, name: str) -> EventstreamType:
        """
        Remove segment synthetic events from eventstream.

        Parameters
        ----------
            %(DropSegment.parameters)s

        Returns
        -------
        EventstreamType
            Eventstream with removed segment.
        """

        from retentioneering.data_processors_lib import DropSegment, DropSegmentParams
        from retentioneering.preprocessing_graph import PreprocessingGraph
        from retentioneering.preprocessing_graph.nodes import EventsNode

        p = PreprocessingGraph(source_stream=self)  # type: ignore
        node = EventsNode(processor=DropSegment(params=DropSegmentParams(name=name)))  # type: ignore
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
