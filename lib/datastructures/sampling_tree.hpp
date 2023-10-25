#ifndef SAMPLING_TREE_H_
#define SAMPLING_TREE_H_

#include <vector>

/**
 * The tree datastructure outlined in Section 6 of our paper.
 *
 * TODO documentation.
 */
class sampling_tree {
public:
private:
  unsigned long m_size;
  std::shared_ptr<std::vector<double>> m_dists_ptr;
  std::vector<double> m_tree;
};

#endif // SAMPLING_TREE_H_
