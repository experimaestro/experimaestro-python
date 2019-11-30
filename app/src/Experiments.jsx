// @flow

import React, { Component } from 'react'
import { connect } from 'react-redux'
import { type State } from './reducer'


type DispatchProps = {
}
type Props = { ...DispatchProps,
    experiment: string
}

class Experiments extends Component<Props> {
    render() {
        // let { experiment } = this.props;
        return <div>
            </div>;
    }
}

const mapStateToProps = (state: State) => ({
    experiment: state.experiment
})
export default connect(mapStateToProps, {  })(Experiments);