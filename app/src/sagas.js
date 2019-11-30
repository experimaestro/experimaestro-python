// @flow

import "regenerator-runtime/runtime";
import { takeEvery, all } from 'redux-saga/effects'
import { toast } from 'react-toastify'

import client from './client'

/// Retrieve the seasons
function* refreshExperimentsSaga(action) : any {
  if (action.payload) {
    // We are connected
    yield client.send({type: "refresh" });
  }
}

function serverError(action) : any {
  toast("Server error: " + action.payload, { type: "error" })
}

export default function* rootSaga() : any {
    yield all([
      takeEvery("CONNECTED", refreshExperimentsSaga),
      takeEvery("SERVER_ERROR", serverError)
    ])
  }
