// @flow

import React, { Component } from 'react'
import { connect } from 'react-redux'

import Clipboard from 'react-clipboard.js';
import { toast } from 'react-toastify';
import { withStyles } from '@material-ui/core/styles';

import TextField from '@material-ui/core/TextField';

import { type State, type Jobs } from './reducer'
import confirm from 'util/confirm';
import client from 'client'
import Theme from 'theme'
import TaskDetail from './TaskDetail'

type StateProps = {
    jobs: Jobs
};

type Props = StateProps;
type OwnState = {
    jobId: ?string;
    searchtask: string;
    searchtagstring: string;
    searchtags: Array<{tag: string, value: string}>
}

const styles = theme => ({
  container: {
    display: 'flex',
    flexWrap: 'wrap',
  },
  textField: {
    marginLeft: theme.spacing.unit,
    marginRight: theme.spacing.unit,
    width: 200,
    padding: 10
  },
  dense: {
    marginTop: 19,
  },
  menu: {
    width: 200,
  },
});


class Tasks extends Component<Props, OwnState> {
    state = {
        searchtask: "",
        searchtagstring: "",
        searchtags: [],
        jobId: null
    }

    kill = (jobId: string) => {
        confirm('Are you sure to kill this job?').then(() => {
            client.send({ type: "kill", payload: jobId}, "cannot kill job " + jobId)
        }, () => {
            toast.info("Action cancelled");
        });
    }

    details = (jobId: ?string) => {
        if (jobId) {
            client.send({ type: "details", payload: jobId}, "cannot get details for job " + jobId);
        }
        this.setState({ ...this.state, jobId: jobId});
    }

    handleChange = name => event => {
        this.setState({ ...this.state, 
            [name]: event.target.value
        });
    };

    handleTagChange = event => {
        let tag = event.target.value;
        let re = /(\S+):(?:([^"]\S*)|"([^"]+)")\s*/g;
        var match = [];
        var tags = [];
        while ((match = re.exec(tag)) !== null) {
            // $FlowFixMe
            tags.push({ tag: match[1], value: match[2] });
        }
        this.setState({ ...this.state,
            searchtagstring: tag,
            searchtags: tags
        })
    }

    render() {
        let { jobs } = this.props;
        let { searchtask, searchtags, jobId } = this.state;

        if (jobId) {
            let job = jobs.byId[jobId];
            return <TaskDetail job={job} handleClose={() => this.setState({...this.state, jobId: null})}/>;
        }

        return <div id="resources">
            <Theme>
            <div className="search">
                <TextField
                    label="Task"
                    className="textField"
                    value={this.state.searchtask}
                    onChange={this.handleChange('searchtask')}                
                    margin="normal"
                    helperText="Task contains..."
                />
                <TextField
                    label="Tags"
                    className="textField"
                    value={this.state.searchtagstring}
                    onChange={this.handleTagChange}
                    margin="normal"
                    helperText="Search tags (format tag:value)"
                />

            </div>
            {
            jobs.ids.map(jobId => {
                let job = jobs.byId[jobId];

                if (searchtask !== "" && job.taskId.search(searchtask) === -1) {
                    return null;
                }

                mainloop: for(let {tag, value} of searchtags) {
                    for(let tv of job.tags) {
                        if (tv[0].search(tag) !== -1 && tv[1].toString().search(value) !== -1)
                            continue mainloop;
                    }
                    return null;
                }

                return <div className="resource" key={jobId}>
                    {
                        job.status === "running" ?
                        <React.Fragment>
                            <span className="status progressbar-container" title={`${job.progress*100}%`}>
                                <span style={{right: `${(1-job.progress)*100}%`}} className="progressbar"></span><div className="status-running">{job.status}</div>
                            </span> 
                            <i className="fa fa-skull-crossbones action" onClick={() => this.kill(jobId) }/>
                        </React.Fragment>
                        :
                        <span className={`status status-${job.status}`}>{job.status}</span>
                    }
                    <i className="fas fa-eye action" title="Details" onClick={() => this.details(jobId)}/>
                    <span className="job-id"><Clipboard className="clipboard" data-clipboard-text={`${job.taskId}/${job.jobId}`} onSuccess={() => toast.success(`Job path copied`)}>{job.taskId}</Clipboard></span>
                    {
                        job.tags.map((tv) => {
                            return <span key={tv[0]} className="tag">
                                <span className="name">{tv[0]}</span>
                                <span className="value">{tv[1]}</span>
                            </span>
                        })
                    }
                </div>
            })
        }</Theme></div>;
    }
}

const mapStateToProps = (state: State) : StateProps => ({
    jobs: state.jobs
})
export default connect(mapStateToProps)(withStyles(styles)(Tasks));