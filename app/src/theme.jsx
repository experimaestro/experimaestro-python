import React from 'react';

import lightBaseTheme from 'material-ui/styles/baseThemes/lightBaseTheme';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import getMuiTheme from 'material-ui/styles/getMuiTheme';

// import injectTapEventPlugin from 'react-tap-event-plugin';
// injectTapEventPlugin();

const Theme = (props) => (
  <MuiThemeProvider muiTheme={getMuiTheme(lightBaseTheme)}>
    <React.Fragment>{ props.children }</React.Fragment>
  </MuiThemeProvider>
);

export default Theme;