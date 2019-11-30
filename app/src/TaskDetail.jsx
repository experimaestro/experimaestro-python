// @flow

import React, { type Node } from 'react';

import { withStyles } from '@material-ui/core/styles';
// import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import ListItemText from '@material-ui/core/ListItemText';
import ListItem from '@material-ui/core/ListItem';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
// import CloseIcon from '@material-ui/icons/Close';
import Slide from '@material-ui/core/Slide';

import { DateTime } from 'luxon'
import Clipboard from 'react-clipboard.js';
import { toast } from 'react-toastify';

import { type Job } from './reducer'

const styles = {
  appBar: {
    position: 'relative',
  },
  flex: {
    flex: 1,
  },
};

function Transition(props: any) {
  return <Slide direction="up" {...props} />;
}

type Props = {
    job: Job,
    classes: any,
    handleClose: void => void
}

const listitem = (text: string, value: Node) => {
    return (
        <ListItem>
             
                <ListItemText primary={text} secondary={<Clipboard className="clipboard" style={{textAlign: "left"}} data-clipboard-text={value} onSuccess={() => toast.success(`Value copied`)}>{value}</Clipboard>} /> 
    </ListItem>
    )

}

class TaskDetail extends React.Component<Props, { open: boolean }> {
  state = {
    open: false,
  };

  render() {
    const { classes, job } = this.props;

    return (
      <div>
        <Dialog
          fullScreen
          open={true}
          onClose={this.props.handleClose}
          TransitionComponent={Transition}
        >
          <AppBar className={classes.appBar}>
            <Toolbar>
              <IconButton color="inherit" onClick={this.props.handleClose} aria-label="Close">
                <i className="far fa-window-close"/>
              </IconButton>
              <Typography variant="h6" color="inherit" className={classes.flex}>
                Task {job.taskId}
              </Typography>
            </Toolbar>
          </AppBar>
          <List>
            {listitem("Status", job.status)}
            {listitem("Path", job.locator)}
            <Divider />
            {listitem("Submitted", DateTime.fromMillis(1000 * job.submitted).toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS))}         
            {listitem("Start", job.start)}
            {listitem("End", job.end)}
            {listitem("Tags", job.tags)}
          </List>
        </Dialog>
      </div>
    );
  }
}

export default withStyles(styles)(TaskDetail);
