from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.eventstream.types import AddSegmentType, EventstreamType
from retentioneering.utils.doc_substitution import docstrings


class RenameSegmentHelperMixin:
    @docstrings.with_indent(12)
    @time_performance(
        scope="rename_segment",
        event_name="helper",
        event_value="combine",
    )
    def rename_segment(self, old_name: str, new_name: str) -> EventstreamType:
        """
        Rename segment for synthetic eventstream events.

        Parameters
        ----------
            %(RenameSegment.parameters)s

        Returns
        -------
        EventstreamType
            Eventstream with renamed segment.
        """

        from retentioneering.data_processors_lib import (
            RenameSegment,
            RenameSegmentParams,
        )
        from retentioneering.preprocessing_graph import PreprocessingGraph
        from retentioneering.preprocessing_graph.nodes import EventsNode

        p = PreprocessingGraph(source_stream=self)  # type: ignore
        node = EventsNode(processor=RenameSegment(params=RenameSegmentParams(old_name=old_name, new_name=new_name)))  # type: ignore
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
