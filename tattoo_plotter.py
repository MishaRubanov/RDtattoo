
    def plot_side_by_side(
        self,
        title1: str = "molecule a",
        title2: str = "molecule b",
    ) -> tuple[Figure, matplotlib.axes.Axes]:
        """
        Plot two images side by side.

        Args:
            array1 (FloatArrayType): First image array.
            array2 (FloatArrayType): Second image array.
            title1 (str): Title for the first image.
            title2 (str): Title for the second image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(self.a, cmap=cmc.oslo)
        axes[0].set_title(title1)
        axes[0].axis("off")  # Hide axis

        axes[1].imshow(self.b, cmap=cmc.lajolla)
        axes[1].set_title(title2)
        axes[1].axis("off")  # Hide axis

        plt.tight_layout()
        plt.show()
        return fig, axes


def draw(self, ax: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]):
        ax[0].clear()
        ax[1].clear()
        assert isinstance(self.a, np.ndarray), "self.a must be a numpy array"
        assert isinstance(self.b, np.ndarray), "self.b must be a numpy array"

        ax[0].imshow(X=self.a, cmap="jet")  # type: ignore[reportUnknownMemberType]
        ax[1].imshow(self.b, cmap="brg")  # type: ignore[reportUnknownMemberType]

        ax[0].set_title("A, t = {:.2f}".format(self.t))  # type: ignore[reportUnknownMemberType]
        ax[1].set_title("B, t = {:.2f}".format(self.t))  # type: ignore[reportUnknownMemberType]

    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # type: ignore[reportUnknownMemberType]
        return fig, ax

    def plot_evolution_outcome(self, filename: Path, n_steps: int):
        """
        Evolves and save the outcome of evolving the system for n_steps
        """
        fig, ax = self.initialise_figure()

        for _ in range(n_steps):
            self.update()

        self.draw(ax)
        fig.savefig(filename)  # type: ignore[reportUnknownMemberType]
        plt.close()

def plot_side_by_side_animation(self, n_frames: int, interval: int = 100):

        fig = go.Figure()

        frames = []
        for i in range(n_frames):
            for _ in range(self.steps):
                self.update()  # Update the simulator state
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=self.a,
                        colorscale="Inferno",
                        zmin=0,
                        zmax=1,
                        showscale=False,
                        xaxis="x",
                        yaxis="y",
                    ),
                    go.Heatmap(
                        z=self.b,
                        colorscale="Inferno",
                        zmin=0,
                        zmax=1,
                        showscale=False,
                        xaxis="x2",
                        yaxis="y2",
                    ),
                ],
                name=f"frame_{i}",
            )
            frames.append(frame)

        fig.add_trace(
            go.Heatmap(
                z=self.a,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Heatmap(
                z=self.b,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x2",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Side by Side Animation",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": interval, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{k}"],
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": interval},
                                },
                            ],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k in range(n_frames)
                    ],
                }
            ],
            xaxis=dict(domain=[0.0, 0.45]),
            yaxis=dict(domain=[0.0, 0.9]),
            xaxis2=dict(domain=[0.55, 1.0]),
            yaxis2=dict(domain=[0.0, 0.9]),
            showlegend=False,
        )

        fig.frames = frames
        fig.show()

    def generate_animation_with_sliders(
        self,
        title: str = "Molecule A",
        alpha_range: tuple = (-1.0, 1.0),
        n_frames: int = 100,
        interval: int = 100,
    ) -> None:
        """
        Generate an animation for molecule A with two sliders: one for alpha and one for time evolution using Plotly.

        Args:
            title (str): Title for the image.
            alpha_range (tuple): Range of values for the alpha slider.
            n_frames (int): Number of frames in the animation.
            interval (int): Interval between frames in milliseconds.
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        frames = []
        for frame_index in range(n_frames):
            # Update alpha linearly within given range
            alpha = np.linspace(alpha_range[0], alpha_range[1], n_frames)[frame_index]
            self.alpha = alpha
            self.update()  # Update the simulator state

            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=self.a, colorscale="Inferno", zmin=0, zmax=1, showscale=True
                    )
                ],
                name=f"frame_{frame_index}",
            )

            frames.append(frame)

        fig.add_trace(
            go.Heatmap(
                z=self.a,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                showscale=True,
            )
        )

        fig.update_layout(
            title=title,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Time:",
                        "xanchor": "right",
                    },
                    "transition": {"duration": interval, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{k}"],
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": interval},
                                },
                            ],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k in range(n_frames)
                    ],
                },
                {
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Alpha:",
                        "xanchor": "right",
                    },
                    "transition": {"duration": interval, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 100},
                    "len": 0.9,
                    "x": 0.1,
                    "y": -0.1,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{k}"],
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": interval},
                                },
                            ],
                            "label": "{:.2f}".format(alpha),
                            "method": "animate",
                        }
                        for k, alpha in enumerate(
                            np.linspace(alpha_range[0], alpha_range[1], n_frames)
                        )
                    ],
                },
            ],
        )

        fig.frames = frames
        fig.show()
