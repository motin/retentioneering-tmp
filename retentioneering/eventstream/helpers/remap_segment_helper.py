from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.eventstream.types import AddSegmentType, EventstreamType
from retentioneering.utils.doc_substitution import docstrings


class RemapSegmentHelperMixin:
    @docstrings.with_indent(12)
    @time_performance(
        scope="remap_segment",
        event_name="helper",
        event_value="combine",
    )
    def remap_segment(self, name: str, mapping: dict) -> EventstreamType:
        """
        Rename segment in synthetic eventstream events.

        Parameters
        ----------
            %(RemapSegment.parameters)s

        Returns
        -------
        EventstreamType
            Eventstream with remapped segment values.
        """

        from retentioneering.data_processors_lib import RemapSegment, RemapSegmentParams
        from retentioneering.preprocessing_graph import PreprocessingGraph
        from retentioneering.preprocessing_graph.nodes import EventsNode

        p = PreprocessingGraph(source_stream=self)  # type: ignore
        node = EventsNode(processor=RemapSegment(params=RemapSegmentParams(name=name, mapping=mapping)))  # type: ignore
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
