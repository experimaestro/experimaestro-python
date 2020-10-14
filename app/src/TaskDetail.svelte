<script>
  export let classes
  export let job
</script>    

<div>
  <Dialog
    fullScreen
    open={true}
    onClose={this.props.handleClose}
    TransitionComponent={Transition}
  >
    <AppBar class={classes.appBar}>
      <Toolbar>
        <IconButton color="inherit" onClick={this.props.handleClose} aria-label="Close">
          <i class="far fa-window-close"/>
        </IconButton>
        <Typography variant="h6" color="inherit" class={classes.flex}>
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
