#include "bounding_box.cuh"
bool map_contains(std::map<float, int> map, float key)
{
    auto search = map.find(key);
    return search != map.end();
}

void new_bounding_box(std::vector<std::vector<size_t>>& boundingboxes, size_t w,
                      size_t h)
{
    std::vector<size_t> bbox;
    bbox.push_back(w);
    bbox.push_back(h);
    bbox.push_back(w);
    bbox.push_back(h);
    boundingboxes.push_back(bbox);
}

void get_bounding_boxes(matrixImage<float>* mat_in,
                        std::vector<std::vector<size_t>>& boundingboxes)
{
    spdlog::info("getting bounding boxes");
    std::map<float, int> met_labels;
    int nb_labels = 0;
    for (size_t w = 0; w < mat_in->width; w++)
    {
        for (size_t h = 0; h < mat_in->height; h++)
        {
            float label = *(mat_in->at(w, h));
            if (label == 0.f)
                continue;
            if (!map_contains(met_labels, label))
            {
                new_bounding_box(boundingboxes, w, h);
                met_labels[label] = nb_labels;
                nb_labels++;
            }
            int bbox_idx = met_labels[label];
            std::vector<size_t> bbox = boundingboxes[bbox_idx];
            if (bbox[0] > w)
                boundingboxes[bbox_idx][0] = w; // upper left x
            if (bbox[2] < w)
                boundingboxes[bbox_idx][2] = w; // lower right x
            if (bbox[1] > h)
                boundingboxes[bbox_idx][1] = h; // upper left y
            if (bbox[3] < h)
                boundingboxes[bbox_idx][3] = h; // lower right y
        }
    }
    for (int i = 0; i < nb_labels; i++)
    {
        boundingboxes[i][2] -= boundingboxes[i][0];
        boundingboxes[i][3] -= boundingboxes[i][1];
    }
}
