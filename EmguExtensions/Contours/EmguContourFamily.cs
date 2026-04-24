/*
*   MIT License
*
*   Copyright (c) 2026 Tiago Conceição
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in all
*   copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*   SOFTWARE.
*/

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using Emgu.CV.Util;

namespace EmguExtensions;

/// <summary>
/// A tree node representing a contour and its nested children in the hierarchy,
/// where even-depth nodes are positive (solid) contours and odd-depth nodes are negative (hole) contours.
/// </summary>
public class EmguContourFamily : IReadOnlyList<EmguContourFamily>
{
    /// <summary>
    /// The list of child contour families, representing nested contours (holes within solids, solids within holes, etc.).
    /// </summary>
    private readonly List<EmguContourFamily> _children = [];

    /// <summary>
    /// Gets the index of this contour within the contour list.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the <see cref="EmguContour"/> data associated with this family node.
    /// </summary>
    public EmguContour Self { get; }

    /// <summary>
    /// Gets the depth level in the contour hierarchy (0 = external/root).
    /// </summary>
    public int Depth { get; }

    /// <summary>
    /// Gets the root contour, which is the topmost external contour in the hierarchy.
    /// </summary>
    public EmguContourFamily Root
    {
        get
        {
            var current = this;

            while (!current.IsExternal)
            {
                current = current.Parent;
            }

            return current;
        }
    }

    /// <summary>
    /// Gets the parent contour family, or <see langword="null"/> if this is an external/root contour.
    /// </summary>
    public EmguContourFamily? Parent { get; }

    /// <summary>
    /// Gets the parent contour index, or -1 if this is an external/root contour.
    /// </summary>
    public int ParentIndex => Parent?.Index ?? -1;

    /// <summary>
    /// Gets whether the contour is a root/external contour (depth 0).
    /// </summary>
    [MemberNotNullWhen(false, nameof(Parent))]
    public bool IsExternal => Depth == 0;

    /// <summary>
    /// Gets whether the contour is a positive (solid) contour, i.e. at an even depth level.
    /// </summary>
    public bool IsPositive => Depth % 2 == 0;

    /// <summary>
    /// Gets whether the contour is a negative (hole) contour, i.e. at an odd depth level.
    /// </summary>
    public bool IsNegative => !IsPositive;

    /// <summary>
    /// Gets whether this contour has any children contours.
    /// </summary>
    public bool HasChildren => _children.Count > 0;

    /// <inheritdoc />
    public int Count => _children.Count;

    /// <inheritdoc />
    public EmguContourFamily this[int index] => _children[index];

    /// <summary>
    /// Gets the total solid area enclosed in this contour, including children contours,
    /// which is the sum of all positive areas minus the sum of all negative areas.
    /// </summary>
    /// <remarks>This property is lazily computed and cached on first access.
    /// Call only on external contours / root for meaningful results.</remarks>
    public double TotalSolidArea
    {
        get
        {
            if (double.IsNaN(field))
            {
                double area = 0;

                foreach (var family in TraverseTree())
                {
                    if (family.IsPositive) area += family.Self.Area;
                    else area -= family.Self.Area;
                }

                field = area;
            }

            return field;
        }
    } = double.NaN;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmguContourFamily"/> class.
    /// </summary>
    /// <param name="index">The index of this contour within the contour list.</param>
    /// <param name="depth">The depth level in the contour hierarchy (0 = external/root).</param>
    /// <param name="self">The contour data for this family node.</param>
    /// <param name="parent">The parent contour family, or <see langword="null"/> for root contours.</param>
    public EmguContourFamily(int index, int depth, EmguContour self, EmguContourFamily? parent)
    {
        Index = index;
        Depth = depth;
        Self = self;
        Parent = parent;
    }

    /// <summary>
    /// Traverses the contour tree in a breadth-first manner.
    /// </summary>
    /// <returns>An enumerable of all contour families in the tree, starting from this node.</returns>
    public IEnumerable<EmguContourFamily> TraverseTree()
    {
        var queue = new Queue<EmguContourFamily>();
        queue.Enqueue(this);

        while (queue.Count > 0)
        {
            var currentFamily = queue.Dequeue();
            yield return currentFamily;
            foreach (var child in currentFamily)
            {
                queue.Enqueue(child);
            }
        }
    }

    /// <summary>
    /// Traverses the contour tree in a breadth-first manner, returning only the <see cref="EmguContour"/> data.
    /// </summary>
    /// <returns>An enumerable of all contours in the tree, starting from this node.</returns>
    public IEnumerable<EmguContour> TraverseTreeAsEmguContour()
    {
        return TraverseTree().Select(family => family.Self);
    }

    /// <summary>
    /// Gets the contours of this family and its children down to the last level.
    /// </summary>
    /// <remarks>Always dispose the returned <see cref="VectorOfVectorOfPoint"/>.</remarks>
    /// <returns>A new instance of <see cref="VectorOfVectorOfPoint"/> holding all contour points.</returns>
    public VectorOfVectorOfPoint ToVectorOfVectorOfPoint()
    {
        var contours = new VectorOfVectorOfPoint();

        foreach (var family in TraverseTree())
        {
            contours.Push(family.Self.Vector);
        }

        return contours;
    }

    /// <summary>
    /// Gets the contours of this family and its children down to the last level.
    /// </summary>
    /// <returns>An array of all <see cref="EmguContour"/> instances in the tree.</returns>
    public EmguContour[] ToEmguContourArray()
    {
        return TraverseTreeAsEmguContour().ToArray();
    }

    /// <summary>
    /// Adds a child contour to this family node.
    /// </summary>
    /// <param name="child">The child contour family to add.</param>
    internal void AddChild(EmguContourFamily child) => _children.Add(child);

    /// <inheritdoc />
    public IEnumerator<EmguContourFamily> GetEnumerator() => _children.GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
