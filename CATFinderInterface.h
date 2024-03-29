#include <cdc/dataobjects/CDCHit.h>
#include <cdc/geometry/CDCGeometryPar.h>
#include <framework/datastore/DataStore.h>
#include <framework/datastore/StoreArray.h>
#include <framework/logging/Logger.h>
#include <framework/pybasf2/PyDBObj.h>
#include <tracking/dataobjects/RecoTrack.h>
#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <vector>

namespace Belle2 {

  class CATFinderInterface {

  private:

    CDC::CDCGeometryPar& cdcgeometrypar = CDC::CDCGeometryPar::Instance();
    StoreArray<CDCHit> m_CDCHits;
    StoreArray<RecoTrack> m_RecoTracks;

    PyObject* py_cdchit_x_temp;
    PyObject* py_cdchit_y_temp;
    PyObject* py_cdchit_tdc_temp;
    PyObject* py_cdchit_adc_temp;
    PyObject* py_cdchit_sl_temp;
    PyObject* py_cdchit_l_temp;

    void preparePyLists(int dataSize)
    {
      // Create new lists with the correct size
      py_cdchit_x_temp = PyList_New(dataSize);
      py_cdchit_y_temp = PyList_New(dataSize);
      py_cdchit_tdc_temp = PyList_New(dataSize);
      py_cdchit_adc_temp = PyList_New(dataSize);
      py_cdchit_sl_temp = PyList_New(dataSize);
      py_cdchit_l_temp = PyList_New(dataSize);
    }

  public:

    CATFinderInterface()
    {
      m_CDCHits.isRequired();
      m_RecoTracks.isRequired();
      m_CDCHits.registerRelationTo(m_RecoTracks);
    }

    void preprocess()
    {

      int cdcHitAmount = m_CDCHits.getEntries();
      preparePyLists(cdcHitAmount); // Clears PyLists and reserves memory space

      for (int i = 0; i < cdcHitAmount; i++) {
        const CDCHit& cdchit = *m_CDCHits[i];

        auto m_clayer = cdchit.getICLayer();
        auto m_wire = cdchit.getIWire();
        auto m_wirepos = cdcgeometrypar.c_Aligned;
        double m_adc = cdchit.getADCCount();

        // Scale TDC and ADC
        double tdc = (static_cast<double>(cdchit.getTDCCount()) - 4100) / 1100;

        if (m_adc > 600) m_adc = 600;

        double adc = m_adc / 600;

        Belle2::B2Vector3 pos_forward = cdcgeometrypar.wireForwardPosition(m_clayer, m_wire, m_wirepos);
        Belle2::B2Vector3 pos_backward = cdcgeometrypar.wireBackwardPosition(m_clayer, m_wire, m_wirepos);

        // Scale x and y position
        PyList_SetItem(py_cdchit_x_temp, i, PyFloat_FromDouble((pos_forward.x() + pos_backward.x()) / 200));
        PyList_SetItem(py_cdchit_y_temp, i, PyFloat_FromDouble((pos_forward.y() + pos_backward.y()) / 200));

        PyList_SetItem(py_cdchit_tdc_temp, i, PyFloat_FromDouble(tdc));
        PyList_SetItem(py_cdchit_adc_temp, i, PyFloat_FromDouble(adc));

        PyList_SetItem(py_cdchit_sl_temp, i, PyFloat_FromDouble(static_cast<double>(cdchit.getISuperLayer()) / 10));
        PyList_SetItem(py_cdchit_l_temp, i, PyFloat_FromDouble(static_cast<double>(cdchit.getILayer()) / 56));
      }

    }

    void postprocess(PyObject* con_point_x, PyObject* con_point_y, PyObject* con_point_z,
                     PyObject* con_point_px, PyObject* con_point_py, PyObject* con_point_pz,
                     PyObject* con_point_vx, PyObject* con_point_vy, PyObject* con_point_vz,
                     PyObject* coord_x, PyObject* coord_y, PyObject* coord_z,
                     PyObject* cdc_hit_index, PyObject* hit_distance)
    {

      std::vector<double> vec_con_point_x = convertPyListToDoubleVector(con_point_x);
      std::vector<double> vec_con_point_y = convertPyListToDoubleVector(con_point_y);
      std::vector<double> vec_con_point_z = convertPyListToDoubleVector(con_point_z);

      std::vector<double> vec_con_point_px = convertPyListToDoubleVector(con_point_px);
      std::vector<double> vec_con_point_py = convertPyListToDoubleVector(con_point_py);
      std::vector<double> vec_con_point_pz = convertPyListToDoubleVector(con_point_pz);

      std::vector<double> vec_con_point_vx = convertPyListToDoubleVector(con_point_vx);
      std::vector<double> vec_con_point_vy = convertPyListToDoubleVector(con_point_vy);
      std::vector<double> vec_con_point_vz = convertPyListToDoubleVector(con_point_vz);

      std::vector<double> vec_coord_x = convertPyListToDoubleVector(coord_x);
      std::vector<double> vec_coord_y = convertPyListToDoubleVector(coord_y);
      std::vector<double> vec_coord_z = convertPyListToDoubleVector(coord_z);

      std::vector<long> vec_cdc_hit_index = convertPyListToLongVector(cdc_hit_index);

      double d_hit_distance = PyFloat_AsDouble(hit_distance);

      for (size_t con_point = 0; con_point < vec_con_point_x.size(); con_point++) {

        std::vector<double> r;
        const size_t size = vec_coord_x.size();
        r.reserve(size);

        for (size_t i = 0; i < size; i++) {
          r.push_back(std::hypot(vec_con_point_x[con_point] - vec_coord_x[i],
                                 vec_con_point_y[con_point] - vec_coord_y[i],
                                 vec_con_point_z[con_point] - vec_coord_z[i]));
        }

        std::vector<int> vec_cdc_hit_indices;
        
        for (size_t i = 0; i < r.size(); ++i) {
          if (r[i] < d_hit_distance) {
            vec_cdc_hit_indices.push_back(vec_cdc_hit_index[i]);
          }
        }

        std::vector<double> momentum = {vec_con_point_px[con_point],
                                        vec_con_point_py[con_point],
                                        vec_con_point_pz[con_point]
                                       };

        std::vector<double> position = {vec_con_point_vx[con_point] * 100.0,
                                        vec_con_point_vy[con_point] * 100.0,
                                        vec_con_point_vz[con_point] * 100.0
                                       };

        auto recotrack = m_RecoTracks.appendNew();
        recotrack->setPositionAndMomentum(ROOT::Math::XYZVector(position[0], position[1], position[2]),
                                          ROOT::Math::XYZVector(momentum[0], momentum[1], momentum[2]));
        recotrack->setChargeSeed(1);

        auto seed_cov = TMatrixDSym(6);
        seed_cov[0][0] = 1e-2;
        seed_cov[1][1] = 1e-2;
        seed_cov[2][2] = 4e-2;
        seed_cov[3][3] = 0.01e-2;
        seed_cov[4][4] = 0.01e-2;
        seed_cov[5][5] = 0.04e-2;

        recotrack->setSeedCovariance(seed_cov);

        for (int j : vec_cdc_hit_indices) {
          recotrack->addCDCHit(m_CDCHits[j], j);
        }

      }

    }

    std::vector<double> convertPyListToDoubleVector(PyObject* pyList)
    {
      std::vector<double> result;

      if (!PyList_Check(pyList)) {
        PyErr_SetString(PyExc_TypeError, "Object is not a list.");
        return result;
      }

      const Py_ssize_t size = PyList_Size(pyList);
      result.reserve(size);

      for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* listItem = PyList_GetItem(pyList, i);

        if (!PyFloat_Check(listItem)) {
          PyErr_SetString(PyExc_TypeError, "List must contain only floats.");
          result.clear();
          break;
        }

        result.emplace_back(PyFloat_AsDouble(listItem));

      }

      return result;

    }

    std::vector<long> convertPyListToLongVector(PyObject* pyList)
    {
      std::vector<long> result;

      if (!PyList_Check(pyList)) {
        PyErr_SetString(PyExc_TypeError, "Object is not a list.");
        return result;
      }

      const Py_ssize_t size = PyList_Size(pyList);
      result.reserve(size);

      for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* listItem = PyList_GetItem(pyList, i);

        if (!PyLong_Check(listItem)) {
          PyErr_SetString(PyExc_TypeError, "List must contain only longs.");
          result.clear();
          break;
        }

        result.emplace_back(PyLong_AsLong(listItem));

      }

      return result;

    }

    void cleanUp()
    {
      Py_DecRef(py_cdchit_x_temp);
      Py_DecRef(py_cdchit_y_temp);
      Py_DecRef(py_cdchit_tdc_temp);
      Py_DecRef(py_cdchit_adc_temp);
    }

    PyObject* getCDCHitXTemp()
    {
      return py_cdchit_x_temp;
    }

    PyObject* getCDCHitYTemp()
    {
      return py_cdchit_y_temp;
    }

    PyObject* getCDCHitTdcTemp()
    {
      return py_cdchit_tdc_temp;
    }

    PyObject* getCDCHitAdcTemp()
    {
      return py_cdchit_adc_temp;
    }

    PyObject* getCDCHitSlTemp()
    {
      return py_cdchit_sl_temp;
    }

    PyObject* getCDCHitLTemp()
    {
      return py_cdchit_l_temp;
    }

  };

}
