from typing import Optional

import samna


def majority_readout_filter(
    feature_count: int,
    default_feature: Optional[int] = None,
    detection_threshold: int = 0,
    threshold_low: int = 0,
    threshold_high: Optional[int] = None,
):
    """
    The default reaodut filter of samna's visualizer counts the total
    number of events received per timestep to decide whether a detection
    should be made or not.

    The filter defined here allows for an additional `detection_threshold`
    parameter which is compared to the number of spikes of the most
    active class.
    In other words, for a class to be detected, there needs to be
    a minimum number of spikes for this class.
    """

    jit_src = f"""
using InputT = speck2f::event::Spike;
using OutputT = ui::Event;
using ReadoutT = ui::Readout;

template<typename Spike>
class CustomMajorityReadout : public iris::FilterInterface<std::shared_ptr<const std::vector<Spike>>, std::shared_ptr<const std::vector<OutputT>>> {{
private:
    int featureCount = {feature_count};
    uint32_t defaultFeature = {default_feature if default_feature is not None else feature_count};
    int detectionThreshold = {detection_threshold};
    int thresholdLow = {threshold_low};
    int thresholdHigh = {threshold_high if threshold_high is not None else "std::numeric_limits<int>::max()"};

public:
    void apply() override
    {{
        while (const auto maybeSpikesPtr = this->receiveInput()) {{
            if (0 == featureCount) {{
                return;
            }}

            auto outputCollection = std::make_shared<std::vector<OutputT>>();
            if ((*maybeSpikesPtr)->size() >= thresholdLow && (*maybeSpikesPtr)->size() <= thresholdHigh) {{
                std::unordered_map<uint32_t, int> sum; // feature -> count
                int maxCount = 0;
                uint32_t maxCountFeature = 0;
                int maxCountNum = 0;

                for (const auto& spike : (**maybeSpikesPtr)) {{
                    sum[spike.feature]++;
                }}

                for (const auto& [feature, count] : sum) {{
                    if (feature >= featureCount) {{
                        continue;
                    }}

                    if (count > maxCount) {{
                        maxCount = count;
                        maxCountFeature = feature;
                        maxCountNum = 1;
                    }}
                    else if (count == maxCount) {{
                        maxCountNum++;
                    }}
                }}

                if (maxCount > detectionThreshold && 1 == maxCountNum) {{
                    outputCollection->emplace_back(ReadoutT{{maxCountFeature}});
                }}
                else {{
                    outputCollection->emplace_back(ReadoutT{{defaultFeature}});
                }}
            }}
            else {{
                outputCollection->emplace_back(ReadoutT{{defaultFeature}});
            }}
            this->forwardResult(std::move(outputCollection));
        }}
    }}
}};
"""
    return samna.graph.JitFilter("CustomMajorityReadout", jit_src)
